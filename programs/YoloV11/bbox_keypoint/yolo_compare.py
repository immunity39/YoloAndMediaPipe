# compare_bbox_keypoint.py
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os

# モデル読み込み
model_bbox = YOLO("best_bbox.pt")   # bboxモデル
model_kpt  = YOLO("best_kpt.pt")    # keypointモデル

# 評価対象（動画 or 画像フォルダ）
SOURCE = "test_video.mp4"  # or "./test_images"

# 結果保存用
results_csv = []

cap = cv2.VideoCapture(SOURCE) if os.path.exists(SOURCE) else None
frame_idx = 0

while True:
    if cap:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        # フォルダから静止画を順番に読む場合
        img_files = sorted(os.listdir(SOURCE))
        if frame_idx >= len(img_files):
            break
        frame = cv2.imread(os.path.join(SOURCE, img_files[frame_idx]))

    if frame is None:
        break

    frame_idx += 1
    h, w = frame.shape[:2]

    # --- bbox model ---
    res_bbox = model_bbox.predict(source=frame, imgsz=640, conf=0.4, verbose=False)
    bbox_tip = None
    if len(res_bbox) > 0 and len(res_bbox[0].boxes) > 0:
        box = res_bbox[0].boxes[0]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        bbox_tip = ((x1+x2)/2, (y1+y2)/2)
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.circle(frame, (int(bbox_tip[0]), int(bbox_tip[1])), 4, (0,255,0), -1)
        cv2.putText(frame, "bbox", (int(x1),int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # --- keypoint model ---
    res_kpt = model_kpt.predict(source=frame, imgsz=640, conf=0.4, verbose=False)
    kpt_tip = None
    if len(res_kpt) > 0 and hasattr(res_kpt[0], "keypoints") and res_kpt[0].keypoints is not None:
        kpts = res_kpt[0].keypoints.xy
        if kpts is not None and len(kpts) > 0:
            xk, yk = kpts[0][0]
            kpt_tip = (float(xk), float(yk))
            cv2.circle(frame, (int(xk), int(yk)), 5, (0,0,255), -1)
            cv2.putText(frame, "keypoint", (int(xk), int(yk)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # --- 結果保存 ---
    if bbox_tip or kpt_tip:
        results_csv.append({
            "frame": frame_idx,
            "bbox_x": bbox_tip[0] if bbox_tip else np.nan,
            "bbox_y": bbox_tip[1] if bbox_tip else np.nan,
            "kpt_x":  kpt_tip[0]  if kpt_tip  else np.nan,
            "kpt_y":  kpt_tip[1]  if kpt_tip  else np.nan,
        })

    cv2.imshow("BBox vs Keypoint", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# CSV出力
pd.DataFrame(results_csv).to_csv("pred_bbox_kpt.csv", index=False)
print("結果を pred_bbox_kpt.csv に保存しました。")
