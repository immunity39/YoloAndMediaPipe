import cv2
import numpy as np
from cv2 import aruco
import mediapipe as mp

# ---- User params ----
CAM_ID = 0
BOARD_MARKER_IDS = [4, 5, 7, 6]  # [top-left, top-right, bottom-left, bottom-right]
MARKER_LENGTH = 0.04
MARKER_GAP_X = 0.20
MARKER_GAP_Y = 0.15
ARUCO_DICT = aruco.DICT_4X4_50
AXIS_LEN = 0.03
USE_SMOOTHING = True
SMOOTH_ALPHA = 0.3

FLIP_FRAME = False

TOUCH_Z_THRESH = 0.01
TOUCH_XY_THRESH = 0.03
TOUCH_CONFIRM_FRAMES = 3
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
    """指定座標(x, y)を中心としたマーカの4隅座標を返す"""
    half = MARKER_LENGTH / 2
    return np.array([
        [x - half, y + half, z],
        [x + half, y + half, z],
        [x + half, y - half, z],
        [x - half, y - half, z]
    ], dtype=np.float32)

def build_board():
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

def line_plane_intersection(plane_point, plane_normal, ray_origin, ray_dir):
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        return None
    d = np.dot(plane_point - ray_origin, plane_normal) / denom
    if d < 0:
        return None
    return ray_origin + d * ray_dir

def project_point(pt3, K, dist):
    proj, _ = cv2.projectPoints(np.array([pt3]), np.zeros(3), np.zeros(3), K, dist)
    return tuple(proj.ravel().astype(int))

def camera_to_board_local(hit, R, tvec):
    return R.T @ (hit - tvec.reshape(3,))

def main():
    K, dist = load_camera_calibration()
    cap = cv2.VideoCapture(CAM_ID)
    dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = build_board()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    ema_center = None
    ema_normal = None

    print("=== Hand–Board Contact Detection ===")
    print("Press 'q' to quit")

    ids_num =-1 

    while True:
        ret, frame = cap.read()
        if not ret: break

        # if needed, flip frame
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, dict_obj)
        plane_ok = False
        center = None
        normal = None
        retval = 0
        rvec = None
        tvec = None
        R = None

        if ids is not None and len(ids) > 0:
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
            if retval and retval > 0:
                plane_ok = True
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXIS_LEN)
                T = rvec_tvec_to_transform(rvec, tvec)
                center = T[:3, 3]
                normal = T[:3, 2]
                R = T[:3, :3]
        else:
            retval, rvec, tvec = None, None, None

        # if ids is not None and len(ids) != ids_num:
        #     ids_num = len(ids)
        #     print ("count ids: ", len(ids) if ids is not None else 0)

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


            # project center and axis for visualization
            proj_center, _ = cv2.projectPoints(np.array([center]), np.zeros(3), np.zeros(3), K, dist)
            pcx, pcy = proj_center.ravel().astype(int)
            cv2.circle(frame, (pcx, pcy), 6, (255,0,0), -1)

            # draw axis
            end = center + normal * 0.05
            proj_end, _ = cv2.projectPoints(np.array([end]), np.zeros(3), np.zeros(3), K, dist)
            epx, epy = proj_end.ravel().astype(int)
            cv2.arrowedLine(frame, (pcx, pcy), (epx, epy), (0,255,0), 2, tipLength=0.3)

            cv2.putText(frame, f"Center (m): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            cv2.putText(frame, f"Normal: {normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)


            # compute pixel radius for XY threshold visualization:
            # axis_x in camera coords is R[:,0]
            axis_x_cam = R[:,0]
            pt_axis = center + axis_x_cam * TOUCH_XY_THRESH
            proj_axis, _ = cv2.projectPoints(np.array([pt_axis]), np.zeros(3), np.zeros(3), K, dist)
            ax_px, ay_px = proj_axis.ravel().astype(int)
            # radius in pixels from center projection
            radius_px = int(np.hypot(ax_px - pcx, ay_px - pcy))
            # draw allowed contact circle (visual)
            cv2.circle(frame, (pcx, pcy), max(10, radius_px), (255, 200, 0), 2)

        else:
            cv2.putText(frame, "Board not detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # detect hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fingertip_world = None
        frame_touch = False


        if results.multi_hand_landmarks and plane_ok:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 画像座標系での指先 (index_finger_tip = 8)
                tip = hand_landmarks.landmark[8]
                u = tip.x * frame.shape[1]
                v = tip.y * frame.shape[0]

                # カメラ座標系へのレイを生成
                pt_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
                ray_dir = pt_cam / np.linalg.norm(pt_cam)
                ray_origin = np.zeros(3)

                # intersect with board plane
                hit = line_plane_intersection(center, normal, ray_origin, ray_dir)

                if hit is None:
                    cv2.putText(frame, "No intersection", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    continue
                else:
                    fingertip_world = hit  # in camera coords on the board plane

                    # convert to board local coordinates (x,y: in-plane, z: along local normal)
                    local = camera_to_board_local(hit, R, tvec)
                    local_x, local_y, local_z = local[0], local[1], local[2]

                    # planar distance (in meters) from center: use x,y components
                    planar_dist = np.linalg.norm(local[:2])
                    # perpendicular distance (signed) from plane center along board z
                    perp_dist = local_z  # can be positive/negative depending on axis orientation
                    abs_perp_dist = abs(perp_dist)

                    # raw per-frame check: both conditions must be satisfied
                    if (abs_perp_dist <= TOUCH_Z_THRESH) and (planar_dist <= TOUCH_XY_THRESH):
                        frame_touch = True
                    else:
                        frame_touch = False

                    # draw hit point projection
                    hit_px = project_point(hit, K, dist)
                    cv2.circle(frame, hit_px, 4, (0, 255, 255), -1)
                    # draw debug numbers
                    cv2.putText(frame, f"z:{abs_perp_dist*1000:.1f}mm xy:{planar_dist*1000:.1f}mm",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                break

        # confirmation logic (debounce)
        if frame_touch:
            touch_count += 1
        else:
            touch_count = 0
            touch_confirmed = False

        if touch_count >= TOUCH_CONFIRM_FRAMES:
            touch_confirmed = True

        if touch_confirmed:
            cv2.putText(frame, "TOUCH CONFIRMED", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        # draw markers
        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Hand-Board Contact", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
