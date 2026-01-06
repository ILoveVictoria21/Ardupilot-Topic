from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import sys

# ======================= 基本設定 =======================
CONNECTION_STRING = "tcp:127.0.0.1:5762"   # 依實際情況修改

TAKEOFF_ALT = 10.0           # 起飛/巡航高度（公尺）
ARRIVAL_RADIUS = 8.0         # 到達航點判定半徑（公尺）

# 各航點經緯度（高度都使用 TAKEOFF_ALT）
START_LAT = 24.146149072846978
START_LON = 120.66119631695354

WP1_LAT = 24.14616738763474
WP1_LON = 120.66163097070067

WP2_LAT = 24.14602625843496
WP2_LON = 120.66162986871478

WP3_LAT = 24.144331730828934
WP3_LON = 120.66167923629328


# ======================= 共用工具函數 =======================
def wait_for(cond_fn, timeout, poll=0.5, desc="條件"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if cond_fn():
                return True
        except:
            pass
        time.sleep(poll)
    print(f"等待超時：{desc}")
    return False


def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6378137.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def safe_set_mode(vehicle, mode_name, timeout=8):
    try:
        vehicle.mode = VehicleMode(mode_name)
        return wait_for(lambda: vehicle.mode.name == mode_name, timeout, desc=f"切換到 {mode_name}")
    except Exception as e:
        print(f"切換模式失敗 {mode_name}：", e)
        return False


def arm_vehicle(vehicle, timeout=20):
    print("準備解鎖...")
    
    # 等待 is_armable
    if not wait_for(lambda: vehicle.is_armable, 15, desc="is_armable"):
        print("警告：is_armable 仍為 False，嘗試強制解鎖（僅限模擬器/除錯）")
    
    # 確保在 GUIDED 模式
    if not safe_set_mode(vehicle, "GUIDED"):
        print("無法進入 GUIDED 模式，無法繼續")
        return False
    
    vehicle.armed = True
    
    return wait_for(lambda: vehicle.armed, timeout, desc="解鎖成功")


def simple_takeoff_wait(vehicle, target_alt, timeout=60):
    print(f"起飛至相對高度 {target_alt}m ...")
    try:
        vehicle.simple_takeoff(target_alt)
    except Exception as e:
        print("simple_takeoff 指令失敗:", e)
        return False
    
    def reached():
        alt = vehicle.location.global_relative_frame.alt
        return alt is not None and alt >= target_alt * 0.95
    
    return wait_for(reached, timeout, poll=1, desc="到達目標高度")


def goto_wait(vehicle, lat, lon, altitude, radius=ARRIVAL_RADIUS, timeout=120):
    print(f"前往目標 → {lat:.8f}, {lon:.8f} @ {altitude}m")
    loc = LocationGlobalRelative(lat, lon, altitude)
    
    try:
        vehicle.simple_goto(loc)
    except Exception as e:
        print("simple_goto 失敗:", e)
    
    def reached():
        try:
            cur = vehicle.location.global_relative_frame
            if cur.lat is None:
                return False
            dist = haversine_distance_m(cur.lat, cur.lon, lat, lon)
            alt_ok = abs(cur.alt - altitude) < 2.0 if cur.alt is not None else False
            return dist <= radius and alt_ok
        except:
            return False
    
    return wait_for(reached, timeout, poll=1, desc="到達航點")


def land_and_disarm(vehicle, timeout=90):
    print("開始自動降落（LAND 模式）...")
    if not safe_set_mode(vehicle, "LAND", timeout=10):
        print("無法進入 LAND 模式！")
        return False
    
    def really_landed():
        alt = vehicle.location.global_relative_frame.alt
        return (alt is not None and alt <= 0.5) and (not vehicle.armed)
    
    success = wait_for(really_landed, timeout, poll=1.5, desc="安全著陸並上鎖")
    if success:
        print("已安全著陸並上鎖")
    else:
        print("降落超時，嘗試強制 disarm...")
        try:
            vehicle.armed = False
        except:
            pass
    return success


# ======================= 主流程 =======================
def main():
    print("連接中...", CONNECTION_STRING)
    try:
        vehicle = connect(CONNECTION_STRING, wait_ready=True, timeout=60)
    except Exception as e:
        print("連接失敗:", e)
        sys.exit(1)

    print("Home 位置：", vehicle.home_location)

    # 顯示航點資訊
    print("\n任務航點：")
    print(f"起點:     {START_LAT:.8f}, {START_LON:.8f}")
    print(f"航點1:    {WP1_LAT:.8f}, {WP1_LON:.8f}")
    print(f"航點2:    {WP2_LAT:.8f}, {WP2_LON:.8f}")
    print(f"航點3:    {WP3_LAT:.8f}, {WP3_LON:.8f}\n")

    # 解鎖
    if not arm_vehicle(vehicle):
        vehicle.close()
        return

    # 起飛
    if not simple_takeoff_wait(vehicle, TAKEOFF_ALT, 70):
        print("起飛失敗 → 嘗試直接降落")
        land_and_disarm(vehicle)
        vehicle.close()
        return

    # 任務開始：先回到起點（確保位置穩定）
    print("回到起點並短暫懸停...")
    if not goto_wait(vehicle, START_LAT, START_LON, TAKEOFF_ALT, timeout=90):
        print("無法回到起點 → 直接降落")
        land_and_disarm(vehicle)
        vehicle.close()
        return
    time.sleep(8)  # 短暫懸停觀察

    # 依序飛往各航點
    waypoints = [
        (WP1_LAT, WP1_LON, "航點一"),
        (WP2_LAT, WP2_LON, "航點二"),
        (WP3_LAT, WP3_LON, "航點三"),
        (START_LAT, START_LON, "返回起點")
    ]

    for lat, lon, name in waypoints:
        print(f"\n=== 前往 {name} ===")
        if not goto_wait(vehicle, lat, lon, TAKEOFF_ALT):
            print(f"{name} 到達失敗 → 直接返航降落")
            break
        time.sleep(3)  # 短暫停留確認

    print("\n任務航線完成，準備降落...")
    time.sleep(2)

    # 安全降落
    land_and_disarm(vehicle)

    print("關閉連線...")
    vehicle.close()


if __name__ == "__main__":
    main()