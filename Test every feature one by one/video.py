import cv2
import time

VIDEO_PATH = r"C:\Users\NBuser\Desktop\KNUS7575.MP4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("無法開啟影片")
    exit()

prev_time = 0
scale = 0.25  # 放大比例

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 放大影像
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # 計算 FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # 標出 FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 顯示影片
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
