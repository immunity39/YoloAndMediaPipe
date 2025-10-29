import os
from ultralytics import YOLO
import cv2
from glob import glob

MODEL_PATH = "yolo_custom.pt"   # あなたのモデル
IMG_DIR = "images"
OUT_DIR = "detections_yolo"
CLASS_NAME = "hand"   # 評価でのクラス名 (GTと一致させる)

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

img_paths = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))

for p in img_paths:
    img = cv2.imread(p)
    res = model(p)  # ultralytics returns results; can use model.predict
    # res is a list-like; take first
    r = res[0]
    # boxes: xyxy, confidence, cls
    boxes = r.boxes  # ultralytics box container
    out_file = os.path.join(OUT_DIR, os.path.basename(p).replace(".jpg", ".txt"))
    with open(out_file, "w") as f:
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()   # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else float(box.conf)
            # class id -> name (if single class, use CLASS_NAME)
            # if multi-class, map box.cls -> class_names[idx]
            line = f"{CLASS_NAME} {conf:.4f} {int(xyxy[0])} {int(xyxy[1])} {int(xyxy[2])} {int(xyxy[3])}\n"
            f.write(line)
    print("Wrote", out_file)
