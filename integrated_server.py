import asyncio
import json
import logging
import os
import ssl
import uuid
import cv2
import fractions
import numpy as np
import math
import time
import threading
from collections import deque
from pathlib import Path
from datetime import datetime
from av import VideoFrame
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from ultralytics import YOLO
import torch

# ==========================================
# 配置參數
# ==========================================
MODEL_PATH = r"C:\Users\LOQ\Downloads\best.pt"
SCREENSHOTS_DIR = r"C:\Users\LOQ\Desktop\webRTC - 複製\screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# YOLO 參數
CONF_THRES = 0.60
IOU_THRES = 0.5
TARGET_CLASS = [0]  # 車輛類別
USE_HALF = True

# 碰撞偵測參數
HFOV = 78  # 水平視角
FLIGHT_HEIGHT = 90  # 假設高度(公尺)
SCALE = 0.5  # 影像縮放比例
BASE_SCALE = 0.5
SCALE_FACTOR = SCALE / BASE_SCALE

DISTANCE_THRESHOLD_PX = int(300 * SCALE_FACTOR)
TOO_CLOSE_THRESHOLD_PX = int(100 * SCALE_FACTOR)
TTC_WARNING_THRES = 0.8
CONSECUTIVE_FRAMES_THRES = 1
MIN_SPEED_THRES = 10.0
ACCEL_THRES = 4.0

# 設備設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")
torch.backends.cudnn.benchmark = True

# 全域變數
current_collision_status = {
    "status": "normal",  # normal, warning, collision
    "message": "",
    "timestamp": ""
}
collision_status_lock = threading.Lock()

# ==========================================
# Vehicle 類別
# ==========================================
class Vehicle:
    def __init__(self, track_id, bbox, meter_per_pixel):
        self.id = track_id
        self.meter_per_pixel = meter_per_pixel
        self.bbox = bbox
        self.center = self._get_center(bbox)
        self.speed_kmh = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = 0.0
        self.center_history = deque(maxlen=10)
        self.center_history.append((self.center, time.time()))
        self.velocity_history = deque(maxlen=10)
        self.last_update_time = time.time()
        self.last_known_bbox = bbox

    def _get_center(self, bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def update(self, bbox):
        curr_time = time.time()
        new_center = self._get_center(bbox)
        
        if len(self.center_history) > 0:
            last_center, last_time = self.center_history[-1]
            dt = curr_time - last_time
            if dt > 0.001:
                disp_px = new_center - last_center
                v_inst = (disp_px * self.meter_per_pixel) / dt
                self.velocity_history.append(v_inst)
        
        self.bbox = bbox
        self.last_known_bbox = bbox
        self.center = new_center
        self.center_history.append((new_center, curr_time))
        self.last_update_time = curr_time
        
        old_velocity = self.velocity
        if self.velocity_history:
            self.velocity = np.mean(self.velocity_history, axis=0)
            raw_speed = np.linalg.norm(self.velocity) * 3.6
            self.speed_kmh = raw_speed if raw_speed >= 2.0 else 0.0
            
            if len(self.center_history) >= 2:
                _, last_time = self.center_history[-2]
                dt = curr_time - last_time
                if dt > 0.001:
                    speed_diff = np.linalg.norm(self.velocity) - np.linalg.norm(old_velocity)
                    self.acceleration = speed_diff / dt

    def predict(self):
        """預測跳幀期間的位置"""
        curr_time = time.time()
        dt = curr_time - self.last_update_time
        
        if dt > 0 and self.meter_per_pixel is not None:
            disp_m = self.velocity * dt
            disp_px = disp_m / self.meter_per_pixel
            
            w = self.last_known_bbox[2] - self.last_known_bbox[0]
            h = self.last_known_bbox[3] - self.last_known_bbox[1]
            
            last_cx, last_cy = self._get_center(self.last_known_bbox)
            new_cx = last_cx + disp_px[0]
            new_cy = last_cy + disp_px[1]
            
            self.bbox = [new_cx - w/2, new_cy - h/2, new_cx + w/2, new_cy + h/2]
            self.center = np.array([new_cx, new_cy])

# ==========================================
# CollisionDetector 類別
# ==========================================
class CollisionDetector:
    def __init__(self):
        self.collision_counter = {}
        self.low_speed_counter = {}

    def get_overlap_metrics(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        interArea = interW * interH
        
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        min_area = min(areaA, areaB)
        overlap_ratio = interArea / min_area if min_area > 0 else 0
        
        is_lateral = interW > interH if interArea > 0 else False
        
        return overlap_ratio, interArea, is_lateral

    def get_trajectory_info(self, v1: Vehicle, v2: Vehicle):
        D = v2.center - v1.center
        dist_m = np.linalg.norm(D) * v1.meter_per_pixel
        V_rel = v2.velocity - v1.velocity
        D_m = D * v1.meter_per_pixel
        dot_prod = np.dot(D_m, V_rel)
        is_approaching = dot_prod < -0.5
        
        norm_v1 = np.linalg.norm(v1.velocity)
        norm_v2 = np.linalg.norm(v2.velocity)
        angle_diff = 0
        if norm_v1 > 0.1 and norm_v2 > 0.1:
            cos_theta = np.dot(v1.velocity, v2.velocity) / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_diff = np.degrees(np.arccos(cos_theta))
        return dist_m, is_approaching, angle_diff

    def check(self, vehicles):
        current_collisions = []
        current_pairs = set()
        vehicle_list = list(vehicles.values())
        num_vehicles = len(vehicle_list)
        
        if num_vehicles < 2:
            return current_collisions
        
        for i in range(num_vehicles):
            v1 = vehicle_list[i]
            for j in range(i+1, num_vehicles):
                v2 = vehicle_list[j]
                key = tuple(sorted((v1.id, v2.id)))
                current_pairs.add(key)

                if v1.speed_kmh < MIN_SPEED_THRES and v2.speed_kmh < MIN_SPEED_THRES:
                    continue

                dist_px = np.linalg.norm(v1.center - v2.center)

                if dist_px > DISTANCE_THRESHOLD_PX:
                    continue
                
                dist_m, is_approaching, angle_diff = self.get_trajectory_info(v1, v2)
                overlap_ratio, overlap_area, is_lateral = self.get_overlap_metrics(v1.bbox, v2.bbox)
                
                # 計算 TTC
                ttc = 99.9
                D = v2.center - v1.center
                D_m = D * v1.meter_per_pixel
                V_rel = v2.velocity - v1.velocity
                dist_m_val = np.linalg.norm(D_m)
                
                if dist_m_val > 0.01:
                    dot_p = np.dot(D_m, V_rel)
                    if dot_p < 0:
                        speed_app = -dot_p / dist_m_val
                        if speed_app > 0.1:
                            ttc = dist_m_val / speed_app

                # 判斷碰撞狀態
                coll_type = ""
                is_collision_candidate = False
                
                if (ttc < TTC_WARNING_THRES and overlap_ratio > IOU_THRES and 
                    v1.speed_kmh >= MIN_SPEED_THRES and v2.speed_kmh >= MIN_SPEED_THRES):
                    coll_type = "COLLISION"
                    is_collision_candidate = True
                elif ttc < TTC_WARNING_THRES:
                    coll_type = "TTC_WARN"
                elif dist_px < TOO_CLOSE_THRESHOLD_PX: 
                    coll_type = "TOO_CLOSE"
                
                if is_collision_candidate:
                    self.collision_counter[key] = self.collision_counter.get(key, 0) + 1
                else:
                    if key in self.collision_counter:
                        self.collision_counter[key] = max(0, self.collision_counter[key] - 1)
                        if self.collision_counter[key] == 0:
                            del self.collision_counter[key]
                
                frames_count = self.collision_counter.get(key, 0)
                
                if coll_type:
                    current_collisions.append((v1, v2, frames_count, coll_type, ttc, overlap_ratio, overlap_area, is_lateral, angle_diff, is_approaching))

        for k in list(self.collision_counter.keys()):
            if k not in current_pairs:
                del self.collision_counter[k]
        for k in list(self.low_speed_counter.keys()):
            if k not in current_pairs:
                del self.low_speed_counter[k]
        return current_collisions

# ==========================================
# VideoDetectionTrack - 整合偵測的影像軌道
# ==========================================
class VideoDetectionTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, camera_id=0, width=1280, height=720):
        super().__init__()
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"無法開啟攝影機 {camera_id}")
        
        # 設定攝影機參數 (盡量接近請求的解析度)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.target_width = width
        self.target_height = height
        
        # 載入 YOLO 模型
        self.model = YOLO(MODEL_PATH)
        self.model.to(device)
        
        # 初始化偵測器
        self.detector = CollisionDetector()
        self.vehicles = {}
        self.meter_per_pixel = None
        self.frame_count = 0
        
        # 時間戳記
        self._timestamp = 0
        self._start = None

    async def recv(self):
        if self._start is None:
            self._start = asyncio.get_event_loop().time()
        
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.camera.read)
        
        if not ret:
            frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        else:
            # 強制縮放到目標解析度
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # 計算 meter_per_pixel (基於當前寬度)
            if self.meter_per_pixel is None:
                # 使用 SCALE_FACTOR 調整計算，因為參數是基於原始設計的 SCALE
                # 這裡我們假設 FLIGHT_HEIGHT 和 HFOV 是固定的
                self.meter_per_pixel = (2 * FLIGHT_HEIGHT * math.tan(math.radians(HFOV / 2))) / self.target_width
            
            # 執行偵測
            frame = await loop.run_in_executor(None, self._process_frame, frame)
        
        # 轉換為 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 建立 VideoFrame
        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = self._timestamp
        new_frame.time_base = fractions.Fraction(1, 30)
        
        self._timestamp += 1
        
        await asyncio.sleep(0.01)
        
        return new_frame

    def _process_frame(self, frame):
        """處理單一影格：偵測、追蹤、繪製"""
        self.frame_count += 1
        
        # 每幀都執行偵測
        results = self.model.track(frame, persist=True, device=device, half=USE_HALF,
                                   verbose=False, classes=TARGET_CLASS, conf=CONF_THRES,
                                   iou=IOU_THRES, imgsz=640)[0]
        
        # 更新車輛資訊
        if results is not None and results.boxes:
            current_ids = set()
            for box in results.boxes:
                if box.id is None:
                    continue
                tid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                bbox = [x1, y1, x2, y2]
                
                if tid not in self.vehicles:
                    self.vehicles[tid] = Vehicle(tid, bbox, self.meter_per_pixel)
                else:
                    self.vehicles[tid].update(bbox)
                current_ids.add(tid)
            
            # 移除消失的車輛
            for tid in list(self.vehicles.keys()):
                if tid not in current_ids:
                    del self.vehicles[tid]
        
        # 檢查碰撞
        collisions = self.detector.check(self.vehicles)
        
        # 更新全域碰撞狀態
        global current_collision_status
        with collision_status_lock:
            if collisions:
                # 找出最嚴重的碰撞
                has_collision = any(coll_type == "COLLISION" for _, _, _, coll_type, _, _, _, _, _, _ in collisions)
                if has_collision:
                    current_collision_status = {
                        "status": "collision",
                        "message": "偵測到碰撞！",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    current_collision_status = {
                        "status": "warning",
                        "message": "警告：車輛過近",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                current_collision_status = {
                    "status": "normal",
                    "message": "正常",
                    "timestamp": datetime.now().isoformat()
                }
        
        # 繪製碰撞框
        collision_ids = set()
        for v1, v2, frames_count, coll_type, ttc, overlap_ratio, overlap_area, is_lateral, angle, approaching in collisions:
            if coll_type == "COLLISION":
                color = (0, 0, 255)  # 紅色
            else:
                color = (0, 165, 255)  # 橙色
            
            v1_bbox = list(map(int, v1.bbox))
            v2_bbox = list(map(int, v2.bbox))
            
            cv2.rectangle(frame, (v1_bbox[0], v1_bbox[1]), (v1_bbox[2], v1_bbox[3]), color, 4)
            cv2.rectangle(frame, (v2_bbox[0], v2_bbox[1]), (v2_bbox[2], v2_bbox[3]), color, 4)
            cv2.putText(frame, f"{coll_type}!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            
            collision_ids.add(v1.id)
            collision_ids.add(v2.id)
            
            # 儲存截圖
            if frames_count == CONSECUTIVE_FRAMES_THRES:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{coll_type}_{timestamp}_ID{v1.id}_ID{v2.id}.jpg"
                filepath = os.path.join(SCREENSHOTS_DIR, filename)
                cv2.imwrite(filepath, frame)
                print(f"已儲存截圖：{filename}")
        
        # 繪製正常車輛框
        for vid, v in self.vehicles.items():
            if vid not in collision_ids:
                bbox = list(map(int, v.bbox))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{v.id} {v.speed_kmh:.1f}km/h", 
                          (bbox[0], bbox[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

    def __del__(self):
        if hasattr(self, 'camera'):
            self.camera.release()

# ==========================================
# WebRTC 處理
# ==========================================
pcs = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    # 讀取解析度參數
    resolution = params.get("resolution", "1280x720")
    try:
        width, height = map(int, resolution.split("x"))
    except ValueError:
        width, height = 1280, 720

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    logger = logging.getLogger("pc")
    logger.info(f"{pc_id} Created for {request.remote} with resolution {width}x{height}")

    try:
        video_track = VideoDetectionTrack(camera_id=0, width=width, height=height)
    except RuntimeError as e:
        logger.error(f"無法開啟攝影機: {str(e)}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": str(e)})
        )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"{pc_id} Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(video_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def get_screenshots(request):
    """取得所有截圖列表"""
    screenshots = []
    for filename in sorted(os.listdir(SCREENSHOTS_DIR), reverse=True):
        if filename.endswith('.jpg'):
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            stat = os.stat(filepath)
            screenshots.append({
                "filename": filename,
                "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size
            })
    return web.json_response(screenshots)

async def get_screenshot(request):
    """取得特定截圖"""
    filename = request.match_info['filename']
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    
    if not os.path.exists(filepath):
        return web.Response(status=404, text="Screenshot not found")
    
    return web.FileResponse(filepath)

async def get_collision_status(request):
    """取得目前碰撞狀態"""
    with collision_status_lock:
        return web.json_response(current_collision_status)

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def index(request):
    content = open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(os.path.dirname(__file__), "client.js"), "r", encoding="utf-8").read()
    return web.Response(content_type="application/javascript", text=content)

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    app.router.add_get("/screenshots", get_screenshots)
    app.router.add_get("/screenshots/{filename}", get_screenshot)
    app.router.add_get("/collision_status", get_collision_status)
    
    print(f"伺服器啟動中...")
    print(f"請開啟瀏覽器訪問: http://localhost:8080")
    print(f"截圖儲存目錄: {SCREENSHOTS_DIR}")
    
    web.run_app(app, host="0.0.0.0", port=8080, access_log=None)
