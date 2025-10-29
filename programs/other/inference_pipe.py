import cv2
import os
from glob import glob
import mediapipe as mp
import numpy as np

IMG_DIR = "images"
OUT_DIR = "detections_mediapipe"
os.makedirs(OUT_DIR, exist_ok=True)
CLASS_NAME = "hand"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

img_paths = sorted(glob(IMG_DIR + "/*.jpg"))
for p in img_paths:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    out_file = os.path.join(OUT_DIR, os.path.basename(p).replace(".jpg", ".txt"))
    with open(out_file, "w") as f:
        if res.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
                # compute bbox in image pixels from landmarks
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min = max(int(min(xs) * w) - 5, 0)
                y_min = max(int(min(ys) * h) - 5, 0)
                x_max = min(int(max(xs) * w) + 5, w-1)
                y_max = min(int(max(ys) * h) + 5, h-1)
                # confidence: use handedness score if available
                conf = 0.0
                if res.multi_handedness and len(res.multi_handedness) > i:
                    conf = res.multi_handedness[i].classification[0].score
                line = f"{CLASS_NAME} {conf:.4f} {x_min} {y_min} {x_max} {y_max}\n"
                f.write(line)
    print("Wrote", out_file)
hands.close()
