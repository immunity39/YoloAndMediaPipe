#!/usr/bin/env python3
# board_center.py
# Estimate board plane and center from 4 ArUco markers (robust: uses estimatePoseBoard or per-marker SVD)

import cv2
import numpy as np
from cv2 import aruco
import time

# ---- User params ----
CAM_ID = 0
BOARD_MARKER_IDS = [4, 5, 6, 7] # the 4 marker IDs you've placed (any order)
MARKER_LENGTH = 0.051           # 5.1 cm (m)
BOARD_WIDTH = 0.252             # nominal board width (m)
BOARD_HIGHT = 0.191             # nominal board height (m)
MARKER_SEP = 0.01               # nominal separation used if needed (m)
ARUCO_DICT = aruco.DICT_4X4_50
# ---------------------

def load_camera_calibration(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("calibration.yaml not found")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def build_board_object_points(marker_length, board_width, board_height):
    hw = marker_length / 2.0
    hh = marker_length / 2.0
    s = marker_length / 2.0

    centers = [
        np.array([-hw,  hh, 0]), # top-left
        np.array([ hw,  hh, 0]), # top-right
        np.array([-hw, -hh, 0]), # bottom-left
        np.array([ hw, -hh, 0])  # bottom-right
    ]

    objp = []
    for c in centers:
        corners = np.array([[-s, s, 0], [ s, s, 0], [ s,-s, 0], [-s,-s, 0]], dtype=np.float32) + c
        objp.append(corners)

    return objp

def estimate_plane_from_markers(corners_list, ids_list, K, dist, dict_obj):
    # For each detected marker, estimateSingleMarkers and collect their 4 corner 3D points
    all_pts = []
    for i, idv in enumerate(ids_list):
        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners_list[i]], MARKER_LENGTH, K, dist)
            rvec = rvecs[0].reshape(3, )
            tvec = tvecs[0].reshape(3, )
            R, _ = cv2.Rodrigues(rvec.reshape(3,1))
            s = MARKER_LENGTH / 2.0
            local = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
            pts_cam = (R @ local.T).T + tvec.reshape(1,3)
            all_pts.append(pts_cam)
        except Exception:
            pass
    if len(all_pts) == 0:
        return None, None, None
    all_pts = np.vstack(all_pts)
    centroid = all_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(all_pts - centroid)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return centroid, normal, all_pts

def main():
    K, dist = load_camera_calibration()
    cap = cv2.VideoCapture(CAM_ID)
    dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = build_board_object_points(MARKER_LENGTH, BOARD_WIDTH, BOARD_HIGHT)
    print("Board object prepared. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, dict_obj)
        plane_ok = False
        centroid = None; normal = None

        if ids is not None and len(ids) >= 1:
            # try estimatePoseBoard first (best)
            try:
                retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
                if retval and retval > 0:
                    plane_ok = True
                    T = np.eye(4)
                    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
                    T[:3,:3] = R
                    T[:3,3] = tvec.reshape(3,)
                    centroid = T[:3,3]
                    normal = T[:3,2]  # z-axis
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.04)
            except Exception:
                pass

            # fallback: SVD from single-marker corners
            if not plane_ok:
                centroid, normal, all_pts = estimate_plane_from_markers(corners, ids.flatten(), K, dist, dict_obj)
                if centroid is not None:
                    plane_ok = True
                    # optional: draw marker corners projected (they are already in camera coords)
                    # project centroid to image
                    proj, _ = cv2.projectPoints(np.array([centroid]), np.zeros(3), np.zeros(3), K, dist)
                    p = tuple(proj.ravel().astype(int))
                    cv2.circle(frame, p, 6, (255,0,0), -1)

        # draw detected markers
        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        if plane_ok:
            # project centroid and show normal vector arrow in image
            proj_c, _ = cv2.projectPoints(np.array([centroid]), np.zeros(3), np.zeros(3), K, dist)
            cpx, cpy = proj_c.ravel().astype(int)
            # draw center
            cv2.circle(frame, (cpx, cpy), 6, (255,0,0), -1)
            # display centroid and normal in camera coords
            cv2.putText(frame, f"Center (m): {centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            cv2.putText(frame, f"Normal: {normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
        else:
            cv2.putText(frame, "Board plane not found", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Board Center Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
