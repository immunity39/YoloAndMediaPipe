import cv2
import numpy as np
import time
from ultralytics import YOLO

# ========== User Settings ==========
CAM_ID = 0
YOLO_MODEL = "yolo11-hand-best.pt"  # 手のポーズ推定対応モデル
CONF_THRESHOLD = 0.3
# ==================================

def main():
    # YOLOモデル読み込み
    model = YOLO(YOLO_MODEL)
    print(f"[INFO] YOLO model loaded: {YOLO_MODEL}")

    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {CAM_ID} not available")

    print("[INFO] Press 'q' to quit.")
    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            break

        # 推論（pose用）
        results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD, max_det=1)
        res = results[0]
        frame_disp = frame.copy()

        # FPS計算
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth != 0 else fps

        # --- 手のキーポイント描画 ---
        if hasattr(res, "keypoints") and res.keypoints is not None:
            kps = getattr(res.keypoints, "xy", None)
            if kps is not None and len(kps) > 0:
                for kp_set in kps:  # 1人分
                    for (x, y) in kp_set:
                        cv2.circle(frame_disp, (int(x), int(y)), 4, (0, 255, 0), -1)
                    # 骨格線も描く
                    for c in getattr(res.keypoints, "connections", []):
                        x1, y1 = kp_set[c[0]]
                        x2, y2 = kp_set[c[1]]
                        cv2.line(frame_disp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                # 最初のキー点を例として表示
                first_kp = tuple(map(int, kps[0][0]))
                cv2.putText(frame_disp, f"({first_kp[0]}, {first_kp[1]})", (first_kp[0]+10, first_kp[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                print(f"[INFO] Keypoints[0]: {kps[0].tolist()}")

        # --- オーバーレイ情報 ---
        cv2.putText(frame_disp, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_disp, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("YOLO Hand Pose Only", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited.")

if __name__ == "__main__":
    main()
