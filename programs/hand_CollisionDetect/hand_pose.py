import cv2, time, math, csv, os
import numpy as np
from ultralytics import YOLO
from cv2 import aruco
import matplotlib.pyplot as plt
from collections import deque

# ========== User params ==========
CAM_ID = 0
YOLO_MODEL = "yolov8n-pose.pt"   # or your trained model path that outputs hand keypoint or grip keypoint
PLANE_ID = 7
PLANE_MARKER_LENGTH = 0.064  # [m]
CONTACT_THRESHOLD = 0.002  # [m]  e.g., 2 mm
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

# plane from aruco marker
def plane_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    normal = R[:,2]  # marker Z as plane normal
    point = tvec.reshape(3,)
    return point, normal

def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance to plane (mm)")
    ax.set_ylim(-5, 50)
    line, = ax.plot([], [], '-o', lw=1, ms=3)
    return fig, ax, line

def main():
    # load models / camera
    model = YOLO(YOLO_MODEL)
    K, dist = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(CAM_ID)
    ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

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
        h, w = frame.shape[:2]

        # detect plane marker
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ar_dict)
        plane_ok = False
        if ids is not None and PLANE_ID in ids.flatten():
            idx = list(ids.flatten()).index(PLANE_ID)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[idx]], PLANE_MARKER_LENGTH, K, dist)
            plane_rvec = rvecs[0].reshape(3); plane_tvec = tvecs[0].reshape(3)
            plane_point, plane_normal = plane_from_rvec_tvec(plane_rvec, plane_tvec)
            plane_ok = True
            cv2.drawFrameAxes(frame, K, dist, plane_rvec, plane_tvec, 0.02)

        # YOLO inference (hand / grip keypoint)
        results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
        res = results[0]
        # obtain keypoint pixel: try res.keypoints.xy or fallback to box center
        hand_px = None
        try:
            # ultralytics returns res.keypoints.xy as array (n_instances, n_keypoints, 2)
            if hasattr(res, "keypoints") and res.keypoints is not None:
                kps = getattr(res.keypoints, "xy", None)
                if kps is None:
                    # older versions may have res.keypoints or need conversion
                    kps = res.keypoints
                if kps is not None and len(kps) > 0:
                    # choose first instance and first keypoint (adapt to your model's keypoint indexing)
                    kp = kps[0][0]  # (x, y)
                    hand_px = (float(kp[0]), float(kp[1]))
        except Exception as e:
            hand_px = None

        # fallback to bbox center if no keypoints
        if hand_px is None:
            try:
                if hasattr(res, "boxes") and len(res.boxes) > 0:
                    box = res.boxes.xyxy[0]  # (x1,y1,x2,y2)
                    x1,y1,x2,y2 = map(float, box)
                    hand_px = ((x1+x2)/2.0, (y1+y2)/2.0)
            except:
                hand_px = None

        tip_cam = None; dist_m = float('nan'); contact = 0
        if hand_px is not None and plane_ok:
            # compute camera ray
            ray_dir = pixel_to_cam_ray(hand_px[0], hand_px[1], K)
            ray_origin = np.array([0.0,0.0,0.0], dtype=np.float64)
            # intersect with plane -> gives 3D point on plane the hand is pointing at
            intersection = ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal)
            if intersection is not None:
                # distance along normal from plane to intersection is zero, but we want distance of hand-supported tip.
                # We assume the "grip" pixel corresponds to a point at or above the plane; use intersection as proxy for where hand aims.
                # If you want "tip offset", add vector along ray direction by offset length.
                tip_cam = intersection  # this is point on plane; if you want grip depth change, adjust
                dist_m = 0.0  # intersection sits on plane
                contact = 1  # hand aiming at plane (candidate)
                # For robust detection, you may want to compute angle between ray and plane normal
                angle = math.degrees(math.acos(abs(np.dot(ray_dir, plane_normal))))
                # optionally require angle small:
                # if angle > 70: contact = 0

        # record & plot
        tnow = time.time()
        times.append(tnow); dists.append(dist_m*1000.0)
        if LOG_CSV:
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([tnow, frame_idx, hand_px[0] if hand_px else "", hand_px[1] if hand_px else "",
                                         tip_cam[0] if tip_cam is not None else "", tip_cam[1] if tip_cam is not None else "",
                                         tip_cam[2] if tip_cam is not None else "", dist_m, contact])

        # plotting
        if len(times) > 1:
            t0 = times[0]; x = [tt - t0 for tt in times]
            y = list(dists)
            line.set_data(x, y)
            ax = plt.gca()
            ax.set_xlim(max(0,x[-1]-10), x[-1]+0.1)
            ax.figure.canvas.draw(); ax.figure.canvas.flush_events(); plt.pause(0.001)

        # visual overlay
        if hand_px is not None:
            cv2.circle(frame, (int(hand_px[0]), int(hand_px[1])), 6, (0,0,255), -1)
        if plane_ok and tip_cam is not None:
            pt2d, _ = cv2.projectPoints(np.array([tip_cam]), np.zeros(3), np.zeros(3), K, dist)
            p = tuple(pt2d.ravel().astype(int))
            cv2.circle(frame, p, 6, (0,255,0), -1)
            cv2.putText(frame, f"CONTACT CANDIDATE", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("YOLO Hand Contact", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
