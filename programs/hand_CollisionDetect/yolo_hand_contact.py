import cv2, time, math, csv, os
import numpy as np
from ultralytics import YOLO
from cv2 import aruco
import matplotlib.pyplot as plt
from collections import deque

# ========== User params ==========
CAM_ID = 0
YOLO_MODEL = "yolo11n-pose.pt"   # or your trained model path that outputs hand keypoint or grip keypoint

# ---------- board marker ---------
BOARD_MARKER_IDS = [4,5,6,7] # top-left, top-right, bottom-left, bottom-right
BOARD_MARKER_LENGTH = 0.051
BOARD_WIDTH = 0.252
BOARD_HEIGHT = 0.191

CONTACT_THRESHOLD = 0.002 # e.g., 2 mm
LOG_CSV = True
LOG_PATH = "yolo_hand_contact_log.csv"
MAX_HISTORY = 300
# =================================

# load camera calibration
def load_camera_calibration(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("calibration.yaml not found")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.reshape(3,)
    return T

def invT(T):
    R=T[:3,:3]; t=T[:3,3]
    Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t
    return Ti

def fit_plane(points):
    """最小二乗平面推定"""
    centroid = points.mean(axis=0)
    _,_,Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return centroid, normal

# pixel -> camera ray (unit vector in camera coords)
def pixel_to_cam_ray(u, v, K):
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    v = np.array([x, y, 1.0], dtype=np.float64)
    v /= np.linalg.norm(v)
    return v

# ray-plane intersection
def ray_plane_intersection(origin, dir_vec, plane_point, plane_normal):
    denom = plane_normal.dot(dir_vec)
    if abs(denom) < 1e-6:
        return None
    t = plane_normal.dot(plane_point - origin) / denom
    if t < 0:
        return None
    return origin + dir_vec * t

def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance to plane (mm)")
    ax.set_ylim(-5, 50)
    line, = ax.plot([], [], '-o', lw=1, ms=3)
    return fig, ax, line

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

def main():
    # load models / camera
    model = YOLO(YOLO_MODEL)
    K, dist = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(CAM_ID)
    ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # build board
    objP = build_board_object_points(BOARD_MARKER_LENGTH, BOARD_WIDTH, BOARD_HEIGHT)
    objP = np.array(objP, dtype=np.float32)

    ids_arr = np.array([[i] for i in BOARD_MARKER_IDS], dtype=np.int32)

    # print(len(objP), len(ids_arr))
    # # → 4 4
    # print(objP[0].shape)
    # # → (4, 3)

    board = aruco.Board(
        objPoints=objP,
        dictionary=ar_dict,
        ids=ids_arr
    )

    # prepare logging & plotting
    if LOG_CSV:
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["time","frame","hand_px_x","hand_px_y","tip_x_m","tip_y_m","tip_z_m","distance_m","contact"])

    times = deque(maxlen=MAX_HISTORY); dists = deque(maxlen=MAX_HISTORY)
    fig, ax, line = init_plot()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ar_dict)

        plane_ok=False
        if ids is not None:
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
            if retval>0:
                plane_ok=True
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)
                T_plane = rvec_tvec_to_T(rvec, tvec)
                plane_point = T_plane[:3,3]
                plane_normal = T_plane[:3,2]

        # YOLOで手（把持位置）推定
        results = model.predict(frame, verbose=False, conf=0.3, max_det=1)
        res = results[0]
        hand_px = None
        if hasattr(res, "keypoints") and res.keypoints is not None:
            kps = getattr(res.keypoints, "xy", None)
            if kps is not None and len(kps)>0:
                kp = kps[0][0]  # 最初のキー点を仮に把持位置とする
                hand_px = (float(kp[0]), float(kp[1]))
        elif hasattr(res, "boxes") and len(res.boxes)>0:
            box = res.boxes.xyxy[0]; x1,y1,x2,y2 = map(float, box)
            hand_px = ((x1+x2)/2.0, (y1+y2)/2.0)

        dist_m = np.nan; contact=False
        if plane_ok and hand_px is not None:
            ray = pixel_to_cam_ray(hand_px[0], hand_px[1], K)
            origin = np.array([0,0,0],dtype=float)
            p_int = ray_plane_intersection(origin, ray, plane_point, plane_normal)
            if p_int is not None:
                # 手先の距離：平面法線方向
                dist_m = abs(np.dot(plane_normal, p_int - plane_point))
                contact = dist_m < CONTACT_THRESHOLD
                pt2d, _ = cv2.projectPoints(np.array([p_int]), np.zeros(3), np.zeros(3), K, dist)
                cv2.circle(frame, tuple(pt2d.ravel().astype(int)), 6, (0,0,255) if contact else (0,255,0), -1)

        # ログとプロット
        now = time.time()
        times.append(now)
        dists.append(dist_m*1000 if not np.isnan(dist_m) else np.nan)
        if LOG_CSV:
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([now, frame_idx, dist_m, int(contact)])

        if len(times)>1:
            t0 = times[0]; x=[t-t0 for t in times]; y=list(dists)
            line.set_data(x,y)
            ax.set_xlim(max(0,x[-1]-10), x[-1]+0.1)
            ax.figure.canvas.draw(); ax.figure.canvas.flush_events(); plt.pause(0.001)

        # オーバーレイ
        if not np.isnan(dist_m):
            cv2.putText(frame, f"Dist: {dist_m*1000:.1f} mm", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if not contact else (0,0,255), 2)
        if contact:
            cv2.putText(frame, "CONTACT", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("YOLO Hand Contact (Board)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()
    plt.ioff()
    print("Finished. CSV:", LOG_PATH)

if __name__ == "__main__":
    main()