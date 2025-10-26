#!/usr/bin/env python3
# mediapipe_board_contact_integration.py
# Uses the board-center estimation approach to compute fingertip <-> board-center distance and contact

import cv2, numpy as np, mediapipe as mp, time, math
from cv2 import aruco

# ----- params -----
CAM_ID = 0
BOARD_MARKER_IDS = [4,5,6,7]
MARKER_LENGTH = 0.051
MARKER_SEP = 0.01
CONTACT_THRESHOLD = 0.002  # meters
FLIP = True
# ------------------

def load_camera_calibration(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("calibration.yaml not found")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def build_gridboard(dict_obj, marker_length, marker_sep):
    board = None
    try:
        board = aruco.GridBoard.create(2, 2, marker_length, marker_sep, dict_obj)
    except AttributeError:
        print("error www")
    return board

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.reshape(3,)
    return T

def pixel_to_cam_ray(u,v,K):
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    x=(u-cx)/fx; y=(v-cy)/fy
    dir = np.array([x,y,1.0])
    dir /= np.linalg.norm(dir)
    return dir

def ray_plane_intersection(origin, dir_vec, plane_point, plane_normal):
    denom = plane_normal.dot(dir_vec)
    if abs(denom) < 1e-9: return None
    t = plane_normal.dot(plane_point - origin) / denom
    if t < 0: return None
    return origin + dir_vec * t

def fit_plane_from_marker_corners(marker_corners_world):
    pts = np.vstack(marker_corners_world)
    centroid = pts.mean(axis=0)
    _,_,Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]; normal /= np.linalg.norm(normal)
    return centroid, normal

def main():
    K, dist = load_camera_calibration()
    cap = cv2.VideoCapture(CAM_ID)
    ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = build_gridboard(ar_dict, MARKER_LENGTH, MARKER_SEP)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    print("Starting integrated fingertip<->board center contact")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if FLIP: frame = cv2.flip(frame, 1)
        h,w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ar_dict)
        plane_ok = False; centroid = None; normal = None

        if ids is not None and len(ids) >= 1:
            # try estimatePoseBoard
            try:
                retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
                if retval and retval > 0:
                    plane_ok = True
                    T = rvec_tvec_to_T(rvec, tvec)
                    centroid = T[:3,3]
                    normal = T[:3,2]
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)
            except Exception:
                # fallback: estimate single markers -> SVD
                marker_pts = []
                for i, idv in enumerate(ids.flatten()):
                    try:
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[i]], MARKER_LENGTH, K, dist)
                        R,_ = cv2.Rodrigues(rvecs[0].reshape(3,1))
                        s = MARKER_LENGTH/2.0
                        local = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float64)
                        pts_cam = (R @ local.T).T + tvecs[0].reshape(1,3)
                        marker_pts.append(pts_cam)
                    except:
                        pass
                if len(marker_pts) > 0:
                    centroid, normal = fit_plane_from_marker_corners(marker_pts)
                    plane_ok = True

        # draw markers
        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        # process hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks and plane_ok:
            # choose first detected hand for simplicity (or loop if multiple)
            for idx, handLms in enumerate(res.multi_hand_landmarks):
                # draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                lm = handLms.landmark[8]  # index finger tip
                px = int(lm.x * w); py = int(lm.y * h)
                cv2.circle(frame, (px, py), 6, (0,255,0), -1)
                # ray-plane intersection
                dir_vec = pixel_to_cam_ray(px, py, K)
                origin = np.array([0.0,0.0,0.0])
                p_int = ray_plane_intersection(origin, dir_vec, centroid, normal)
                if p_int is not None:
                    # distance (Euclidean) between board center and fingertip point on plane
                    d = np.linalg.norm(p_int - centroid)
                    # contact check by signed distance along normal (should be zero ideally)
                    signed = float(np.dot(normal, p_int - centroid))
                    contact = abs(signed) < CONTACT_THRESHOLD
                    # project p_int and centroid to image for visualization
                    proj_tip, _ = cv2.projectPoints(np.array([p_int]), np.zeros(3), np.zeros(3), K, dist)
                    proj_ctr, _ = cv2.projectPoints(np.array([centroid]), np.zeros(3), np.zeros(3), K, dist)
                    tip_px = tuple(proj_tip.ravel().astype(int)); ctr_px = tuple(proj_ctr.ravel().astype(int))
                    cv2.circle(frame, tip_px, 6, (0,255,255), -1)
                    cv2.circle(frame, ctr_px, 6, (255,0,0), -1)
                    # overlay values
                    cv2.putText(frame, f"Centroid: {centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f} m", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.putText(frame, f"Tip3D: {p_int[0]:.3f},{p_int[1]:.3f},{p_int[2]:.3f} m", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.putText(frame, f"Dist center-tip: {d*1000:.1f} mm", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if not contact else (0,0,255), 2)
                    if contact:
                        cv2.putText(frame, "CONTACT", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("MediaPipe fingertip vs board center", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
