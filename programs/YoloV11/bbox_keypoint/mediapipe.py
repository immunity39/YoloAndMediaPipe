import cv2, mediapipe as mp
import numpy as np
from math import sqrt
from utils import load_calib

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# TIP extension length in meters (initial guess; tune by measurement)
EXT_LEN_M = 0.06  # 6cm forward from finger tip in camera ray direction; not exact without depth

def main():
    K, dist = load_calib("calibration.yaml")
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img)
            img_bgr = frame.copy()
            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # choose wrist(0) and index_tip(8)
                    h, w, _ = img_bgr.shape
                    idx_tip = hand_landmarks.landmark[8]
                    wrist = hand_landmarks.landmark[0]
                    # image pixel coordinates
                    px = int(idx_tip.x * w)
                    py = int(idx_tip.y * h)
                    wx = int(wrist.x * w)
                    wy = int(wrist.y * h)
                    cv2.circle(img_bgr, (px,py), 4, (0,0,255), -1)
                    # direction vector in image plane
                    dir_vec = np.array([px - wx, py - wy], dtype=float)
                    norm = np.linalg.norm(dir_vec)
                    if norm > 1e-3:
                        dir_unit = dir_vec / norm
                        # extend in image plane for isualization
                        ex = int(px + dir_unit[0]*80)
                        ey = int(py + dir_unit[1]*80)
                        cv2.line(img_bgr, (px,py),(ex,ey),(255,0,0),2)
                        # Note: this is 2D. To get 3D location, need depth (stereo or known plane).
            cv2.imshow("MP Hands Tip Proxy", img_bgr)
            if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
