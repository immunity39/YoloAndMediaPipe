import cv2, time, csv, os, math, numpy as np
from collections import deque
from cv2 import aruco

# matplotlib backend setting (for some environments)
import matplotlib
matplotlib.use('TkAgg')  # OpenGLを使わない安全な描画
import matplotlib.pyplot as plt

# params (combine previous)
CAM_ID = 0
PLANE_ID = 7; PLANE_MARKER_LENGTH = 0.064
CUBE_IDS = [0,1,2,3]; CUBE_MARKER_LENGTH = 0.0315
CUBE_WIDTH=0.047; CUBE_DEPTH=0.040
TIP_OFFSET = np.array([0.0, -0.020, 0.00], dtype=np.float32)
CONTACT_THRESHOLD = 0.0015
LOG_PATH = "compare_marker_yolo.csv"
MAX_HISTORY=300
MODEL_PATH = "best_kpt.pt"

# try yolov8
try:
    from ultralytics import YOLO
    yolo = YOLO(MODEL_PATH)
    yolo_avail=True
except Exception as e:
    print("YOLO load failed:", e)
    yolo_avail=False

# helper functions (reuse previous small versions)
def load_camera_calibration(file_path="calibration.yaml"):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened(): raise FileNotFoundError("calibration.yaml missing")
    K = fs.getNode("camera_matrix").mat(); dist=fs.getNode("dist_coeff").mat(); fs.release()
    return K, dist

def rvec_tvec_to_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1)); T=np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.reshape(3,); return T
def transform_inverse(T): R=T[:3,:3]; t=T[:3,3]; Tinv=np.eye(4); Tinv[:3,:3]=R.T; Tinv[:3,3]=-R.T@t; return Tinv
def marker_corners_on_face(center,u_vec,v_vec,marker_length):
    half=marker_length/2.0; tl=center - u_vec*half + v_vec*half; tr=center + u_vec*half + v_vec*half
    br=center + u_vec*half - v_vec*half; bl=center - u_vec*half - v_vec*half
    return np.vstack([tl,tr,br,bl]).astype(np.float32)
def build_cube_board(marker_length,width,depth):
    half_w=width/2.0; half_d=depth/2.0
    X=np.array([1,0,0]); Y=np.array([0,1,0]); Z=np.array([0,0,1])
    objPoints=[ marker_corners_on_face(np.array([0,0,+half_d]),X,Y,marker_length),
                marker_corners_on_face(np.array([+half_w,0,0]),-Z,Y,marker_length),
                marker_corners_on_face(np.array([0,0,-half_d]),-X,Y,marker_length),
                marker_corners_on_face(np.array([-half_w,0,0]),Z,Y,marker_length)]
    ids = np.array([[CUBE_IDS[0]],[CUBE_IDS[1]],[CUBE_IDS[2]],[CUBE_IDS[3]]], dtype=np.int32)
    return aruco.Board(objPoints=objPoints, ids=ids, dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_50))

def plane_from_rvec_tvec(rvec,tvec):
    R,_ = cv2.Rodrigues(rvec.reshape(3,1))
    normal = R[:,2]; normal = normal/np.linalg.norm(normal)
    point = tvec.reshape(3,)
    return point, normal

def pixel_to_ray(u,v,K):
    fx=K[0,0]; fy=K[1,1]; cx=K[0,2]; cy=K[1,2]
    x=(u-cx)/fx; y=(v-cy)/fy
    dir_cam = np.array([x,y,1.0],dtype=np.float64); dir_cam /= np.linalg.norm(dir_cam)
    return np.array([0.0,0.0,0.0], dtype=np.float64), dir_cam

def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    denom = plane_normal.dot(ray_dir)
    if abs(denom) < 1e-6: return None
    t = plane_normal.dot(plane_point - ray_origin) / denom
    if t < 0: return None
    return ray_origin + ray_dir * t

# main
def main():
    K, dist = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(CAM_ID)
    cube_board = build_cube_board(CUBE_MARKER_LENGTH, CUBE_WIDTH, CUBE_DEPTH)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # CSV header
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["time","frame","marker_tip_z_m","marker_contact","yolo_tip_z_m","yolo_contact"])

    # plot buffers
    times=deque(maxlen=MAX_HISTORY); marker_h=deque(maxlen=MAX_HISTORY); yolo_h=deque(maxlen=MAX_HISTORY)
    fig, ax = plt.subplots(figsize=(8,4)); plt.ion()
    line_m, = ax.plot([],[], '-o', label='marker height (mm)')
    line_y, = ax.plot([],[], '-o', label='yolo height (mm)')
    ax.legend(); ax.set_xlabel("time (s)"); ax.set_ylabel("height (mm)")
    frame_idx=0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)

        plane_avail=False; cube_avail=False; plane_rvec=plane_tvec=None; cube_rvec=cube_tvec=None
        if ids is not None:
            ids_list = ids.flatten().tolist()
            if PLANE_ID in ids_list:
                idx = ids_list.index(PLANE_ID)
                r_p, t_p, _ = aruco.estimatePoseSingleMarkers([corners[idx]], PLANE_MARKER_LENGTH, K, dist)
                plane_rvec = r_p[0].reshape(3); plane_tvec = t_p[0].reshape(3); plane_avail=True
                cv2.drawFrameAxes(frame, K, dist, plane_rvec, plane_tvec, 0.02)
            retval, rvec_b, tvec_b = aruco.estimatePoseBoard(corners, ids, cube_board, K, dist, None, None)
            if retval>0:
                cube_rvec = rvec_b.reshape(3); cube_tvec = tvec_b.reshape(3); cube_avail=True
                cv2.drawFrameAxes(frame, K, dist, cube_rvec, cube_tvec, 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)

        marker_tip_z = float('nan'); marker_contact=False
        if plane_avail and cube_avail:
            T_plane = rvec_tvec_to_transform(plane_rvec, plane_tvec)
            T_cube = rvec_tvec_to_transform(cube_rvec, cube_tvec)
            tip_local_h = np.hstack([TIP_OFFSET.reshape(3,), 1.0])
            plane_tip = (transform_inverse(T_plane) @ (T_cube @ tip_local_h))[:3]
            marker_tip_z = plane_tip[2]; marker_contact = abs(marker_tip_z) < CONTACT_THRESHOLD

        # YOLO path
        yolo_tip_z = float('nan'); yolo_contact=False
        if yolo_avail:=('yolo' in globals() and yolo_avail):
            results = yolo.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
            res = results[0]
            kp_pix = None
            if hasattr(res, "keypoints") and res.keypoints is not None:
                kps = getattr(res.keypoints, "xy", None)
                if kps is not None and len(kps)>0:
                    kp = kps[0][0]; kp_pix = (float(kp[0]), float(kp[1]))
            if kp_pix is None and hasattr(res, "boxes") and len(res.boxes)>0:
                box = res.boxes.xyxy[0]; x1,y1,x2,y2 = map(float,box); kp_pix = ((x1+x2)/2.0,(y1+y2)/2.0)
            if kp_pix is not None and plane_avail:
                origin, dir_cam = pixel_to_ray(kp_pix[0], kp_pix[1], K)
                ipt = ray_plane_intersection(origin, dir_cam, plane_tvec.reshape(3,), (cv2.Rodrigues(plane_rvec.reshape(3,1))[0])[:,2])
                if ipt is not None:
                    # signed dist
                    R,_ = cv2.Rodrigues(plane_rvec.reshape(3,1)); normal = R[:,2]/np.linalg.norm(R[:,2])
                    signed = normal.dot(ipt - plane_tvec.reshape(3,))
                    yolo_tip_z = signed; yolo_contact = abs(signed) < CONTACT_THRESHOLD

        # logging
        tnow = time.time()
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([tnow, frame_idx,
                                     marker_tip_z if not math.isnan(marker_tip_z) else "",
                                     int(marker_contact),
                                     yolo_tip_z if not math.isnan(yolo_tip_z) else "",
                                     int(yolo_contact)])
        # plotting
        times.append(tnow); marker_h.append(marker_tip_z*1000 if not math.isnan(marker_tip_z) else float('nan'))
        yolo_h.append(yolo_tip_z*1000 if not math.isnan(yolo_tip_z) else float('nan'))
        if len(times)>1:
            t0 = times[0]; x = [tt - t0 for tt in times]
            line_m.set_data(x, list(marker_h)); line_y.set_data(x, list(yolo_h))
            ax.set_xlim(max(0, x[-1]-10), x[-1]+0.1)
            ax.figure.canvas.draw(); ax.figure.canvas.flush_events(); plt.pause(0.001)

        # overlays
        if not math.isnan(marker_tip_z): cv2.putText(frame, f"MarkerZ(mm): {marker_tip_z*1000:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        if not math.isnan(yolo_tip_z): cv2.putText(frame, f"YOLOZ(mm): {yolo_tip_z*1000:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
        if marker_contact: cv2.putText(frame, "CONTACT(marker)", (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        if yolo_contact: cv2.putText(frame, "CONTACT(yolo)", (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow("Compare Marker vs YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); plt.ioff()
    print("Saved:", LOG_PATH)

if __name__=="__main__":
    main()
