from dronekit import connect, VehicleMode, LocationGlobalRelative
import time, math, sys, os
import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")


# 配置
CONNECTION_STRING = "udp:127.0.0.1:14550"
MODEL_PATH = "yolov5s.pt"
VIDEO_DEFAULT = "video2.mp4"
TAKEOFF_ALT = 10.0
WP_A_OFFSET = (50, 0)
WP_B_OFFSET = (0, 50)
ARRIVAL_RADIUS = 8.0
DETECT_FPS = 1
HOVER_SECONDS = 20
CONFIDENCE_THRESHOLD = 0.4


VEHICLE_CLASS_NAMES = set(['car', 'truck', 'bus', 'motorcycle', 'bicycle'])
R = 6378137.0

def get_location_metres(original_location, dNorth, dEast):
    dLat = dNorth / R
    dLon = dEast / (R * math.cos(math.radians(original_location.lat)))
    return original_location.lat + math.degrees(dLat), original_location.lon + math.degrees(dLon)

def haversine_distance_m(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# 模型加载
model = None
def load_model(path, conf=CONFIDENCE_THRESHOLD):
    global model
    try:
        if path and os.path.exists(path):
            print("加载本地权重:", path)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False, trust_repo=True)
        else:
            print("本地权重未找到")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        if hasattr(model, 'conf'):
            model.conf = conf
        print("模型加载完成，置信度阈值 =", conf)
    except Exception as e:
        print("模型加载失败：", e)
        model = None

def video_info(cap):
    import os
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"video fps={fps}, frames={frame_count}, size={width}x{height}")


def detect_vehicle(frame):
    global model
    if model is None:
        return False
    try:
        with torch.no_grad():
            results = model(frame)
        df = results.pandas().xyxy[0]
        if 'name' in df.columns:
            names = [str(x).lower() for x in df['name'].tolist() if x is not None]
            for n in names:
                if n in VEHICLE_CLASS_NAMES:
                    return True
        else:
            if len(df) > 0:
                return True
    except Exception as e:
        print("检测异常（继续）：", e)
    return False


def wait_for_mode(vehicle, mode_name, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if vehicle.mode and vehicle.mode.name == mode_name:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False

def wait_for_landed_and_disarmed(vehicle, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            alt = vehicle.location.global_relative_frame.alt
            armed = vehicle.armed
            print(" 等待降落: alt=%.2f m, armed=%s" % (alt if alt is not None else -1, armed))
            if (alt is not None and alt <= 0.9) and (not armed):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def arm_and_takeoff(vehicle, target_alt):
    # 若在 QRTL 或 RTL，先切 STABILIZE
    try:
        if vehicle.mode and vehicle.mode.name in ("QRTL", "RTL"):
            print("当前处于 QRTL/RTL，先切 STABILIZE 再解锁")
            vehicle.mode = VehicleMode("STABILIZE")
            if not wait_for_mode(vehicle, "STABILIZE", timeout=5):
                print("切 STABILIZE 超时（继续尝试解锁）")
    except Exception as e:
        print("切模式检查异常:", e)

    print("设置 GUIDED 并尝试解锁...")
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(0.5)
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 30:
            raise RuntimeError("解锁超时")
        print(" 等待解锁...")
        time.sleep(0.5)
    print("已解锁，起飞到 %.1f m" % target_alt)
    try:
        vehicle.simple_takeoff(target_alt)
    except Exception as e:
        print("simple_takeoff 调用异常:", e)
    t0 = time.time()
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt is not None:
            print(" 当前相对高度: %.1f m" % alt)
            if alt >= target_alt * 0.9:
                print("到达目标高度")
                break
        if time.time() - t0 > 60:
            raise RuntimeError("起飞超时")
        time.sleep(0.5)

def simple_goto_hold(vehicle, hold_seconds):
    pos = vehicle.location.global_relative_frame
    print("在当前位置悬停,lat=%.7f lon=%.7f alt=%.1f" % (pos.lat, pos.lon, pos.alt if pos.alt else 0.0))
    try:
        vehicle.simple_goto(LocationGlobalRelative(pos.lat, pos.lon, pos.alt))
    except Exception as e:
        print("悬停异常:", e)
    time.sleep(hold_seconds)


def goto_waypoint_and_watch(vehicle, lat, lon, cap):
    print("导航至: %.7f, %.7f" % (lat, lon))
    try:
        vehicle.simple_goto(LocationGlobalRelative(lat, lon, TAKEOFF_ALT))
    except Exception as e:
        print("simple_goto 异常:", e)
    last_detect_time = 0
    fps = cap.get(cv2.CAP_PROP_FPS)     # 视频帧率
    detect_interval = 1.0               # 检测时间间隔
    skip_frames = fps * detect_interval # 每次检测往后跳过多少帧
    frame_index = 0

    while True:
        cur = vehicle.location.global_relative_frame
        if cur.lat is not None:
            d = haversine_distance_m(cur.lat, cur.lon, lat, lon)
            print(" 距目标距离: %.1f m" % d)
            if d <= ARRIVAL_RADIUS:
                print(" 已到达航点,停留10s")
                time.sleep(10)
                return False
            
        if time.time() - last_detect_time >= detect_interval:
            ret, frame = cap.read()
            if not ret:
                print(" 视频结束或读取失败，停止检测当前航点")
                return False
            if detect_vehicle(frame):
                print(" 检测到车辆！")
                return True
            else:
                print("未检测到！")
            last_detect_time = time.time()
            frame_index += skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        time.sleep(0.05)


# 主函数
def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = VIDEO_DEFAULT

    if not os.path.exists(video_path):
        print("未找到视频文件:", video_path); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("打开视频失败:", video_path); return

    load_model(MODEL_PATH)

    print("连接飞控:", CONNECTION_STRING)
    try:
        vehicle = connect(CONNECTION_STRING, wait_ready=True, timeout=60)
    except Exception as e:
        print("连接失败:", e); cap.release(); return
    
    # 打印视频信息
    video_info(cap)

    # 等待 home
    t0 = time.time()
    while not vehicle.location.global_frame.lat and time.time()-t0 < 15:
        print("等待 home 定位...")
        time.sleep(0.5)
    home = vehicle.location.global_frame
    print("Home:", home.lat, home.lon)

    latA, lonA = get_location_metres(home, WP_A_OFFSET[0], WP_A_OFFSET[1])
    latB, lonB = get_location_metres(home, WP_B_OFFSET[0], WP_B_OFFSET[1])
    print("航点 A:", latA, lonA)
    print("航点 B:", latB, lonB)

    # 主任务执行
    try:
        try:
            arm_and_takeoff(vehicle, TAKEOFF_ALT)
        except Exception as e:
            print("起飞失败:", e)
            raise

        # 去 A 点并检测
        foundA = goto_waypoint_and_watch(vehicle, latA, lonA, cap)
        if foundA:
            print("在飞向A点过程中检测到车辆,悬停 %d 秒" % HOVER_SECONDS)
            vehicle.mode = VehicleMode("GUIDED")
            # time.sleep(HOVER_SECONDS)
            simple_goto_hold(vehicle, HOVER_SECONDS)

        # 去 B 点并检测
        if not foundA:
            foundB = goto_waypoint_and_watch(vehicle, latB, lonB, cap)
            if foundB:
                print("在飞向B点过程中检测到车辆,悬停 %d 秒" % HOVER_SECONDS)
                vehicle.mode = VehicleMode("GUIDED")
                simple_goto_hold(vehicle, HOVER_SECONDS)

        # 任务完成，触发 QRTL 返回（四旋翼/quadplane 专用 RTL）
        print("任务结束，切 QRTL 返回并等待降落与 disarm")
        try:
            vehicle.mode = VehicleMode("QRTL")
        except Exception as e:
            print("切 QRTL 异常:", e)

        # 等待降落并 disarm（最长等待 land_timeout 秒）
        landed_disarmed = wait_for_landed_and_disarmed(vehicle, timeout=300)
        if landed_disarmed:
            print("检测到已降落并 disarm")
        else:
            print("等待降落超时")

        # 切回 STABILIZE，确保下次能 arm
        try:
            vehicle.mode = VehicleMode("STABILIZE")
            if wait_for_mode(vehicle, "STABILIZE", timeout=5):
                print("已切换为STABILIZE模式")
            else:
                print("切换STABILIZE超时")
        except Exception as e:
            print("切换STABILIZE 出错:", e)

    finally:
        # 关闭并释放资源
        try:
            vehicle.close()
        except Exception:
            pass
        cap.release()
        print("已释放资源，脚本结束")

if __name__ == "__main__":
    main()
