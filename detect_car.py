from ultralytics import YOLO
import cv2
import math
import torch
import time
import threading
import queue

# ---------------------------
# GPU 設定
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

# ---------------------------
# 影片與模型路徑
# ---------------------------
VIDEO_PATH = r"C:\Users\NBuser\Desktop\KNUS7575.MP4"
MODEL_PATH = r"C:\Users\NBuser\Downloads\best.pt"

# ---------------------------
# 載入模型
# ---------------------------
model = YOLO(MODEL_PATH)
model.to(device)

# ---------------------------
# 儲存前一幀車輛中心點
# ---------------------------
previous_positions = {}  # id : (cx, cy)

# ---------------------------
# IoU 計算
# ---------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union != 0 else 0

# ---------------------------
# TTC 計算
# ---------------------------
def get_ttc(box1, speed1, box2, speed2, meter_per_pixel):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    dist_px = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    dist_m = dist_px * meter_per_pixel
    v_rel = abs(speed1 / 3.6 - speed2 / 3.6)  # km/h -> m/s
    if v_rel > 0:
        ttc = dist_m / v_rel
    else:
        ttc = float('inf')
    return ttc

# ---------------------------
# 影片參數
# ---------------------------
HFOV = 78  # 水平視角 degree
FLIGHT_HEIGHT = 90  # 高度公尺
SCALE = 0.25  # 縮放比例

# ---------------------------
# 車速計算（換算成 km/h）
# ---------------------------
def get_speed_kmh(track_id, bbox, meter_per_pixel, dt):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if track_id in previous_positions:
        px, py = previous_positions[track_id]
        dist_px = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        speed_mps = (dist_px * meter_per_pixel) / dt  # m/s
        speed_kmh = speed_mps * 3.6
    else:
        speed_kmh = 0
    previous_positions[track_id] = (cx, cy)
    return speed_kmh

# ---------------------------
# 影像佇列
# ---------------------------
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

# ---------------------------
# 讀影像線程
# ---------------------------
def read_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE)

    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        frame_queue.put(frame)
    cap.release()

# ---------------------------
# YOLO 推理線程
# ---------------------------
def infer_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put((None, None))
            break
        results = model.track(frame, persist=True, device=device)[0]
        result_queue.put((frame, results))

# ---------------------------
# 啟動線程
# ---------------------------
threading.Thread(target=read_frames, daemon=True).start()
threading.Thread(target=infer_frames, daemon=True).start()

prev_time = 0
frame_width = None
meter_per_pixel = None

# ---------------------------
# 主線程處理畫框與顯示
# ---------------------------
while True:
    frame, results = result_queue.get()
    if frame is None:
        break

    if frame_width is None:
        frame_height, frame_width = frame.shape[:2]
        meter_per_pixel = (2 * FLIGHT_HEIGHT * math.tan(math.radians(HFOV / 2))) / frame_width

    curr_time = time.time()
    dt = curr_time - prev_time if prev_time != 0 else 1/30
    prev_time = curr_time

    cars = []

    # 遍歷車輛
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0])
        track_id = int(box.id) if box.id is not None else None
        if cls == 0 and track_id is not None:
            bbox = [x1, y1, x2, y2]
            speed = get_speed_kmh(track_id, bbox, meter_per_pixel, dt)
            stopped = speed < 2
            cars.append((track_id, bbox, speed, stopped))

            color = (0, 0, 255) if stopped else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{track_id} V:{speed:.1f}km/h", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # IoU 與 TTC 計算
    for i in range(len(cars)):
        for j in range(i + 1, len(cars)):
            id1, box1, s1, st1 = cars[i]
            id2, box2, s2, st2 = cars[j]

            # IoU
            overlap = iou(box1, box2)
            if overlap > 0.1:
                overlap_percent = overlap * 100
                print(f"🚨 車 {id1} 與 車 {id2} 重疊：{overlap_percent:.1f}%")
                cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 165, 255), 2)
                cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 165, 255), 2)
                cv2.putText(frame, f"{overlap_percent:.1f}%", 
                            (int((box1[0]+box2[0])/2), int((box1[1]+box2[1])/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # TTC
            ttc = get_ttc(box1, s1, box2, s2, meter_per_pixel)
            if ttc < 5:  # 5 秒警示
                print(f"⚠️ 車 {id1} 與 車 {id2} 可能碰撞！TTC: {ttc:.2f} 秒")
                cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0,0,255), 3)
                cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0,0,255), 3)
                cv2.putText(frame, f"TTC:{ttc:.1f}s", 
                            (int((box1[0]+box2[0])/2), int((box1[1]+box2[1])/2)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # 計算 FPS
    fps = 1 / dt
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
