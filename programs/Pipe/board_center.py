#!/usr/bin/env python3
# Robust board plane + center estimation even when some markers are occluded

import cv2
import numpy as np
from cv2 import aruco
import time
import math

# ---- User params ----
CAM_ID = 0
BOARD_MARKER_IDS = [4, 5, 6, 7]     # [top-left, top-right, bottom-right, bottom-left]
MARKER_LENGTH = 0.051               # 5.1 cm (m)
BOARD_WIDTH = 0.25                 # board physical width (m)
BOARD_HEIGHT = 0.19                # board physical height (m)
MARKER_SEP = 0.01                   # separation between markers (m)
ARUCO_DICT = aruco.DICT_4X4_50
AXIS_LEN = 0.03
USE_SMOOTHING = True
SMOOTH_ALPHA = 0.3                  # EMA coefficient (0=no smooth, 1=very slow)
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

def transform_inverse(T):
    R = T[:3,:3]; t = T[:3,3]
    Tinv = np.eye(4)
    Tinv[:3,:3] = R.T
    Tinv[:3,3] = -R.T @ t
    return Tinv

def estimate_plane_from_visible_markers(corners, ids, K, dist, marker_length):
    """Fallback: estimate each marker pose, aggregate 3D corner points, fit plane."""
    all_pts = []
    for i, idv in enumerate(ids):
        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[i]], marker_length, K, dist)
            rvec = rvecs[0].reshape(3,)
            tvec = tvecs[0].reshape(3,)
            R, _ = cv2.Rodrigues(rvec.reshape(3,1))
            s = marker_length / 2.0
            local = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
            pts_cam = (R @ local.T).T + tvec.reshape(1,3)
            all_pts.append(pts_cam)
        except Exception:
            continue
    if len(all_pts) == 0:
        return None, None
    all_pts = np.vstack(all_pts)
    centroid = np.mean(all_pts, axis=0)
    _, _, Vt = np.linalg.svd(all_pts - centroid)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return centroid, normal

def main():
    K, dist = load_camera_calibration()
    cap = cv2.VideoCapture(CAM_ID)
    dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT)
    objP = [
        np.array([[-MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
                  [ MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
                  [ MARKER_LENGTH/2,-MARKER_LENGTH/2, 0],
                  [-MARKER_LENGTH/2,-MARKER_LENGTH/2, 0]], dtype=np.float32),
        np.array([[BOARD_WIDTH/2 - MARKER_LENGTH - MARKER_SEP, MARKER_LENGTH/2, 0],
                  [BOARD_WIDTH/2 - MARKER_SEP, MARKER_LENGTH/2, 0],
                  [BOARD_WIDTH/2 - MARKER_SEP, -MARKER_LENGTH/2, 0],
                  [BOARD_WIDTH/2 - MARKER_LENGTH - MARKER_SEP, -MARKER_LENGTH/2, 0]], dtype=np.float32),
        np.array([[BOARD_WIDTH/2 - MARKER_LENGTH - MARKER_SEP, -BOARD_HEIGHT/2 + MARKER_LENGTH + MARKER_SEP, 0],
                  [BOARD_WIDTH/2 - MARKER_SEP, -BOARD_HEIGHT/2 + MARKER_LENGTH + MARKER_SEP, 0],
                  [BOARD_WIDTH/2 - MARKER_SEP, -BOARD_HEIGHT/2 + MARKER_SEP, 0],
                  [BOARD_WIDTH/2 - MARKER_LENGTH - MARKER_SEP, -BOARD_HEIGHT/2 + MARKER_SEP, 0]], dtype=np.float32),
        np.array([[-MARKER_LENGTH/2, -BOARD_HEIGHT/2 + MARKER_LENGTH + MARKER_SEP, 0],
                  [ MARKER_LENGTH/2, -BOARD_HEIGHT/2 + MARKER_LENGTH + MARKER_SEP, 0],
                  [ MARKER_LENGTH/2, -BOARD_HEIGHT/2 + MARKER_SEP, 0],
                  [-MARKER_LENGTH/2, -BOARD_HEIGHT/2 + MARKER_SEP, 0]], dtype=np.float32)
    ]

    board = aruco.Board(
        objPoints=objP,
        ids=np.array([[i] for i in BOARD_MARKER_IDS], dtype=np.int32),
        dictionary=dict_obj
    )

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

        if ids is not None and len(ids) > 0:
            try:
                retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
                if retval and retval > 0:
                    plane_ok = True
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXIS_LEN)
                    T = rvec_tvec_to_transform(rvec, tvec)
                    center = T[:3,3]
                    normal = T[:3,2]
            except Exception:
                plane_ok = False

            if not plane_ok:
                # fallback plane estimation
                center, normal = estimate_plane_from_visible_markers(corners, ids.flatten(), K, dist, MARKER_LENGTH)
                if center is not None:
                    plane_ok = True

        if plane_ok:
            # smoothing (EMA)
            if USE_SMOOTHING:
                if ema_center is None:
                    ema_center = center.copy()
                    ema_normal = normal.copy()
                else:
                    ema_center = SMOOTH_ALPHA * center + (1 - SMOOTH_ALPHA) * ema_center
                    ema_normal = SMOOTH_ALPHA * normal + (1 - SMOOTH_ALPHA) * ema_normal
                    ema_normal /= np.linalg.norm(ema_normal)
                center, normal = ema_center, ema_normal

            # board center projection
            proj, _ = cv2.projectPoints(np.array([center]), np.zeros(3), np.zeros(3), K, dist)
            p = tuple(proj.ravel().astype(int))
            cv2.circle(frame, p, 6, (255,0,0), -1)

            # draw normal vector in image
            end = center + normal * 0.05
            proj_end, _ = cv2.projectPoints(np.array([end]), np.zeros(3), np.zeros(3), K, dist)
            epx, epy = proj_end.ravel().astype(int)
            cv2.arrowedLine(frame, p, (epx, epy), (0,255,0), 2, tipLength=0.3)

            # show text
            cv2.putText(frame, f"Center (m): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            cv2.putText(frame, f"Normal: {normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
        else:
            cv2.putText(frame, "Board not detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # draw detected markers
        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Board Center (Robust)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
