import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 指先の座標を取得（例：人差し指）
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                tip_xy = (int(index_tip.x * w), int(index_tip.y * h))
                cv2.circle(frame, tip_xy, 6, (0,0,255), -1)

                # コテ先を「指方向に延長」して推定（簡易版）
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                dir_vec = np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])
                extended_tip = (int((index_tip.x + dir_vec[0]*0.3)*w), int((index_tip.y + dir_vec[1]*0.3)*h))
                cv2.circle(frame, extended_tip, 4, (255,0,0), -1)

        cv2.imshow("MediaPipe tip estimation", frame)
        if cv2.waitKey(1) == ord('q'):
            break
