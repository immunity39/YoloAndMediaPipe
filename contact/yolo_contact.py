import cv2, math, time, csv, os, numpy as np
from collections import deque
from cv2 import aruco
import matplotlib.pyplot as plt

# ---------- user params ----------
MODEL_PATH = "best_kpt.pt"  # your trained YOLOv8-pose model
CAM_ID = 0
PLANE_ID = 7
PLANE_MARKER_LENGTH = 0.064
TIP_PROXY_SCALE = 0.06  # if YOLO keypoint is on handle, this is not used; for tip keypoint it's direct
CONTACT_THRESHOLD = 0.0015  # m
LOG_CSV = True
LOG_PATH = "yolo_tip_log.csv"
MAX_HISTORY = 300
# ----------------------------------

try:
    from ultralytics import YOLO
    yolo_model = YOLO(MODEL_PATH)
    yolo_available = True
except Exception as e:
    print("Warning: ultralytics model not available or failed to load:", e)
    yolo_available = False

def load_camera_calibration(file_path="calibration.yaml"):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("calibration.yaml not found")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height (mm)")
    ax.set_ylim(-5, 50)
    line, = ax.plot([], [], '-o', lw=1, ms=3)
    return fig, ax, line

# compute plane from rvec,tvec (plane point and normal in camera coords)
def plane_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    # marker plane normal in marker coords is +Z (assuming marker printed on plane facing camera)
    normal = R[:,2]  # third column
    point = tvec.reshape(3,)
    normal = normal / np.linalg.norm(normal)
    return point, normal

# ray-plane intersection in camera coords
def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    denom = plane_normal.dot(ray_dir)
    if abs(denom) < 1e-6:
        return None
    t = plane_normal.dot(plane_point - ray_origin) / denom
    if t < 0:
        return None
    return ray_origin + ray_dir * t

def pixel_to_ray(u, v, K):
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    dir_cam = np.array([x, y, 1.0], dtype=np.float64)
    dir_cam = dir_cam / np.linalg.norm(dir_cam)
    origin = np.array([0.0,0.0,0.0], dtype=np.float64)
    return origin, dir_cam

def main():
    K, dist = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(CAM_ID)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    cube_board = None  # not needed here

    if LOG_CSV:
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["time","frame","tip_cam_x","tip_cam_y","tip_cam_z","contact"])

    # plotting buffers
    times = deque(maxlen=MAX_HISTORY); heights = deque(maxlen=MAX_HISTORY)
    fig, ax, line = init_plot()

    frame_idx = 0; contact_active=False; contact_start=None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)

        plane_available=False; plane_point=None; plane_normal=None
        if ids is not None:
            ids_list = ids.flatten().tolist()
            if PLANE_ID in ids_list:
                idx = ids_list.index(PLANE_ID)
                rvecs_p, tvecs_p, _ = aruco.estimatePoseSingleMarkers([corners[idx]], PLANE_MARKER_LENGTH, K, dist)
                plane_rvec = rvecs_p[0].reshape(3); plane_tvec = tvecs_p[0].reshape(3)
                plane_point, plane_normal = plane_from_rvec_tvec(plane_rvec, plane_tvec)
                plane_available=True
                cv2.drawFrameAxes(frame, K, dist, plane_rvec, plane_tvec, 0.02)
            aruco.drawDetectedMarkers(frame, corners, ids)

        tip_height_mm = np.nan; contact_now=False
        # run yolo if available
        if yolo_available:
            results = yolo_model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
            res = results[0]
            # try to extract keypoints (ultralytics pose outputs)
            keypoint_pixel = None
            if hasattr(res, "keypoints") and res.keypoints is not None:
                # res.keypoints.xy is array of shape (n_instances, n_keypoints, 2)
                kps = getattr(res.keypoints, "xy", None)
                if kps is None:
                    # try other attribute
                    try:
                        kps = res.keypoints
                    except:
                        kps=None
                if kps is not None and len(kps)>0:
                    # choose first instance, first keypoint
                    kp = kps[0][0] if isinstance(kps[0], (list, tuple, np.ndarray)) else kps[0].numpy()[0]
                    keypoint_pixel = (float(kp[0]), float(kp[1]))
            # fallback: use bbox center
            if keypoint_pixel is None and hasattr(res, "boxes") and len(res.boxes)>0:
                box = res.boxes.xyxy[0]
                x1,y1,x2,y2 = map(float, box)
                keypoint_pixel = ((x1+x2)/2.0, (y1+y2)/2.0)

            if keypoint_pixel is not None and plane_available:
                origin, ray_dir = pixel_to_ray(keypoint_pixel[0], keypoint_pixel[1], K)
                ipt = ray_plane_intersection(origin, ray_dir, plane_point, plane_normal)
                if ipt is not None:
                    # compute signed distance along plane normal from plane point: plane_normal dot (ipt - plane_point)
                    signed = plane_normal.dot(ipt - plane_point)
                    tip_height_mm = signed * 1000.0
                    contact_now = abs(signed) < CONTACT_THRESHOLD
                    # draw keypoint and intersection
                    cv2.circle(frame, (int(keypoint_pixel[0]), int(keypoint_pixel[1])), 5, (0,0,255), -1)
                    # project intersection back to image for visualization
                    # optional: draw small circle
            # else no yolo detection or no plane
        # record & plot
        tnow = time.time()
        times.append(tnow)
        heights.append(tip_height_mm)
        if LOG_CSV:
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([tnow, frame_idx, float(ipt[0]) if 'ipt' in locals() else np.nan,
                                        float(ipt[1]) if 'ipt' in locals() else np.nan,
                                        float(ipt[2]) if 'ipt' in locals() else np.nan,
                                        int(contact_now)])
        # contact transitions
        if contact_now and not contact_active:
            contact_active=True; contact_start=tnow; print("[yolo] contact start")
        if not contact_now and contact_active:
            contact_active=False; dur = tnow - (contact_start or tnow); print(f"[yolo] contact end dur={dur:.3f}"); contact_start=None

        # update plot
        if len(times)>1:
            t0 = times[0]; x = [tt - t0 for tt in times]
            line.set_data(x, list(heights))
            ax.set_xlim(max(0, x[-1]-10), x[-1]+0.1)
            ax.figure.canvas.draw(); ax.figure.canvas.flush_events(); plt.pause(0.001)

        # overlay
        if not math.isnan(tip_height_mm):
            cv2.putText(frame, f"YOLO TipZ(mm): {tip_height_mm:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if contact_now else (200,200,200), 2)
            if contact_now: cv2.putText(frame, "CONTACT (YOLO)", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("YOLO Keypoint Contact", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    plt.ioff()
    print("Finished. CSV:", LOG_PATH)

if __name__=="__main__":
    main()
