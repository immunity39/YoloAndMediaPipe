#!/usr/bin/env python3

import cv2
import numpy as np
from cv2 import aruco
import mediapipe as mp
import time

# ---- User params ----
CAM_ID = 0
BOARD_MARKER_IDS = [4, 5, 7, 6]
MARKER_LENGTH = 0.04
MARKER_GAP_X = 0.20
MARKER_GAP_Y = 0.15
ARUCO_DICT = aruco.DICT_4X4_50
AXIS_LEN = 0.03
USE_SMOOTHING = True
SMOOTH_ALPHA = 0.3
TOUCH_THRESHOLD = 0.01  # m
DEBUG = True
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
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = tvec.reshape(3,)
    return T

def make_marker_object_points(x, y, z=0):
    half = MARKER_LENGTH / 2
    return np.array([
        [x - half, y + half, z],
        [x + half, y + half, z],
        [x + half, y - half, z],
        [x - half, y - half, z]
    ], dtype=np.float32)

def build_board_objectpoints_and_map():
    """Return board (for estimatePoseBoard) and a dict mapping id->objectPoints(4x3) for fallback."""
    half_x = MARKER_GAP_X / 2
    half_y = MARKER_GAP_Y / 2
    coords = {
        "tl": (-half_x,  half_y),
        "tr": ( half_x,  half_y),
        "bl": (-half_x, -half_y),
        "br": ( half_x, -half_y)
    }
    id_order = BOARD_MARKER_IDS  # list with length 4
    obj_points = [
        make_marker_object_points(*coords["tl"]),
        make_marker_object_points(*coords["tr"]),
        make_marker_object_points(*coords["bl"]),
        make_marker_object_points(*coords["br"])
    ]
    # aruco.Board expects objPoints as list of 4x3 arrays in same order as ids
    ids_np = np.array([[i] for i in id_order], dtype=np.int32)
    dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.Board(objPoints=obj_points, ids=ids_np, dictionary=dict_obj)

    # create lookup map id -> object points (4x3)
    id_to_obj = {id_order[i]: obj_points[i] for i in range(len(id_order))}
    return board, id_to_obj

def line_plane_intersection(plane_point, plane_normal, ray_origin, ray_dir):
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        return None
    d = np.dot(plane_point - ray_origin, plane_normal) / denom
    if d < 0:
        return None
    return ray_origin + d * ray_dir

def pose_from_marker_corners_fallback(detected_corners, detected_ids, id_to_obj, K, dist):
    """
    Build 3D-2D correspondences from detected markers and run solvePnP.
    detected_corners: list of N arrays (4,1,2) as returned by detectMarkers
    detected_ids: Nx1 array
    id_to_obj: dict id -> (4x3) object points in board frame
    Returns: (success, rvec, tvec)
    """
    if detected_ids is None or len(detected_ids) == 0:
        return False, None, None

    obj_pts_list = []
    img_pts_list = []
    # ensure flatten
    det_ids_flat = detected_ids.flatten()
    for i, det_id in enumerate(det_ids_flat):
        if int(det_id) in id_to_obj:
            # detected_corners[i] has shape (4,1,2)
            img_corners = np.squeeze(detected_corners[i])  # shape (4,2)
            obj_corners = id_to_obj[int(det_id)]          # shape (4,3)
            # Append corresponding points in same order (ArUco uses TL,TR,BR,BL or similar - ensure correct mapping)
            # Our object_points are in order: [(-half, +half), (+half,+half), (+half,-half), (-half,-half)]
            # detectMarkers returns corner ordering consistent with Board usage, so mapping by index is OK.
            for j in range(4):
                obj_pts_list.append(obj_corners[j])
                img_pts_list.append(img_corners[j])
    if len(obj_pts_list) < 4:
        # need at least 4 correspondences (or 3 with solvePnP?), but being conservative require >=4
        return False, None, None

    obj_pts = np.array(obj_pts_list, dtype=np.float32)
    img_pts = np.array(img_pts_list, dtype=np.float32)

    # solvePnP
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return False, None, None
    return True, rvec, tvec

def main():
    K, dist = load_camera_calibration()
    cap = cv2.VideoCapture(CAM_ID)
    dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT)
    board, id_to_obj = build_board_objectpoints_and_map()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    ema_center = None
    ema_normal = None

    print("=== Hand–Board Contact Detection (robust) ===")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(gray, dict_obj)
        if DEBUG:
            print("Detected ids:", None if ids is None else ids.flatten().tolist())

        plane_ok = False
        center = None
        normal = None
        rvec, tvec = None, None

        # Only call estimatePoseBoard if there are detected markers
        if ids is not None and len(ids) > 0:
            try:
                retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
            except cv2.error as e:
                # sometimes estimatePoseBoard throws assertion if ids empty or shape mismatch
                if DEBUG:
                    print("estimatePoseBoard exception:", e)
                retval = 0

            if retval and retval > 0:
                plane_ok = True
                if DEBUG:
                    print("estimatePoseBoard succeeded, retval=", retval)
            else:
                # Fallback: create 3D-2D correspondences from detected markers and call solvePnP
                ok, frvec, ftvec = pose_from_marker_corners_fallback(corners, ids, id_to_obj, K, dist)
                if ok:
                    rvec, tvec = frvec, ftvec
                    plane_ok = True
                    if DEBUG:
                        print("Fallback solvePnP succeeded")
                else:
                    if DEBUG:
                        print("Fallback solvePnP failed (not enough correspondences)")

            if plane_ok:
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXIS_LEN)
                T = rvec_tvec_to_transform(rvec, tvec)
                center = T[:3, 3]
                normal = T[:3, 2]

        # detect hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        touch_detected = False
        fingertip_world = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if plane_ok:
                    tip = hand_landmarks.landmark[8]  # index-tip
                    u = tip.x * frame.shape[1]
                    v = tip.y * frame.shape[0]

                    # form ray in camera coordinates (assuming camera at origin)
                    uv1 = np.array([u, v, 1.0], dtype=np.float64)
                    try:
                        Kinv = np.linalg.inv(K)
                    except np.linalg.LinAlgError:
                        Kinv = np.linalg.pinv(K)
                    pt_cam = Kinv @ uv1
                    ray_dir = pt_cam / np.linalg.norm(pt_cam)
                    ray_origin = np.zeros(3)

                    hit = line_plane_intersection(center, normal, ray_origin, ray_dir)
                    if hit is not None:
                        fingertip_world = hit
                        dist_to_plane = abs(np.dot(normal, (hit - center)))
                        if DEBUG:
                            print(f"hit: {hit}, dist_to_plane: {dist_to_plane:.4f} m")
                        if dist_to_plane < TOUCH_THRESHOLD:
                            touch_detected = True
                            cv2.putText(frame, "TOUCH!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # smoothing + drawing center/normal
        if plane_ok:
            if USE_SMOOTHING:
                if ema_center is None:
                    ema_center = center.copy()
                    ema_normal = normal.copy()
                else:
                    ema_center = SMOOTH_ALPHA * center + (1 - SMOOTH_ALPHA) * ema_center
                    ema_normal = SMOOTH_ALPHA * normal + (1 - SMOOTH_ALPHA) * ema_normal
                    ema_normal /= np.linalg.norm(ema_normal)
                center, normal = ema_center, ema_normal

            proj, _ = cv2.projectPoints(np.array([center]), np.zeros(3), np.zeros(3), K, dist)
            p = tuple(proj.ravel().astype(int))
            cv2.circle(frame, p, 6, (255, 0, 0), -1)

            end = center + normal * 0.05
            proj_end, _ = cv2.projectPoints(np.array([end]), np.zeros(3), np.zeros(3), K, dist)
            epx, epy = proj_end.ravel().astype(int)
            cv2.arrowedLine(frame, p, (epx, epy), (0, 255, 0), 2, tipLength=0.3)

            cv2.putText(frame, f"Center (m): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            cv2.putText(frame, f"Normal: {normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        else:
            cv2.putText(frame, "Board not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Hand-Board Contact (robust)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
