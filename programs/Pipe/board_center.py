#!/usr/bin/env python3
# Robust board plane + center estimation (occlusion-tolerant, configurable grid)

import cv2
import numpy as np
from cv2 import aruco
import math

# ---- User params ----
CAM_ID = 0
BOARD_MARKER_IDS = [4, 5, 7, 6]  # [top-left, top-right, bottom-right, bottom-left]
MARKER_LENGTH = 0.04             # (m)
MARKER_GAP_X = 0.20              # X方向のマーカ間距離 (m)
MARKER_GAP_Y = 0.15              # Y方向のマーカ間距離 (m)
ARUCO_DICT = aruco.DICT_4X4_50
AXIS_LEN = 0.03
USE_SMOOTHING = True
SMOOTH_ALPHA = 0.3
# ---------------------

def load_camera_calibration(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("calibration.yaml not found")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def rvec_tvec_to_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = tvec.reshape(3,)
    return T

def make_marker_object_points(x, y, z=0):
    """指定座標(x, y)を中心としたマーカの4隅座標を返す"""
    half = MARKER_LENGTH / 2
    return np.array([
        [x - half, y + half, z],
        [x + half, y + half, z],
        [x + half, y - half, z],
        [x - half, y - half, z]
    ], dtype=np.float32)

def build_board():
    """
    4つのマーカを平面の四隅に配置。
    配置は:
    (0,0): top-left, (+X): right, (-Y): down
    """
    # ボード中心を原点にした座標系に配置
    half_x = MARKER_GAP_X / 2
    half_y = MARKER_GAP_Y / 2
    coords = {
        "tl": (-half_x,  half_y),
        "tr": ( half_x,  half_y),
        "br": ( half_x, -half_y),
        "bl": (-half_x, -half_y)
    }
    obj_points = [
        make_marker_object_points(*coords["tl"]),
        make_marker_object_points(*coords["tr"]),
        make_marker_object_points(*coords["br"]),
        make_marker_object_points(*coords["bl"])
    ]
    ids = np.array([[i] for i in BOARD_MARKER_IDS], dtype=np.int32)
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    return aruco.Board(objPoints=obj_points, ids=ids, dictionary=dictionary)

def main():
    K, dist = load_camera_calibration()
    cap = cv2.VideoCapture(CAM_ID)
    dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = build_board()

    ema_center = None
    ema_normal = None

    print("=== Board Center Estimation with ArUco (occlusion-tolerant) ===")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, dict_obj)
        plane_ok = False
        center = None
        normal = None
        retval = 0
        rvec = None
        tvec = None

        if ids is not None and len(ids) > 0:
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
            if retval and retval > 0:
                plane_ok = True
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXIS_LEN)
                T = rvec_tvec_to_transform(rvec, tvec)
                center = T[:3,3]
                normal = T[:3,2]

        if plane_ok:
            # smoothing
            if USE_SMOOTHING:
                if ema_center is None:
                    ema_center = center.copy()
                    ema_normal = normal.copy()
                else:
                    ema_center = SMOOTH_ALPHA * center + (1 - SMOOTH_ALPHA) * ema_center
                    ema_normal = SMOOTH_ALPHA * normal + (1 - SMOOTH_ALPHA) * ema_normal
                    ema_normal /= np.linalg.norm(ema_normal)
                center, normal = ema_center, ema_normal

            # draw center + normal
            proj, _ = cv2.projectPoints(np.array([center]), np.zeros(3), np.zeros(3), K, dist)
            p = tuple(proj.ravel().astype(int))
            cv2.circle(frame, p, 6, (255,0,0), -1)

            end = center + normal * 0.05
            proj_end, _ = cv2.projectPoints(np.array([end]), np.zeros(3), np.zeros(3), K, dist)
            epx, epy = proj_end.ravel().astype(int)
            cv2.arrowedLine(frame, p, (epx, epy), (0,255,0), 2, tipLength=0.3)

            # show info
            cv2.putText(frame, f"Center (m): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            cv2.putText(frame, f"Normal: {normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
        else:
            cv2.putText(frame, "Board not detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Board Center", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
