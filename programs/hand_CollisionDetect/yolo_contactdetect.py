import cv2, numpy as np, csv, time, math
from cv2 import aruco
from ultralytics import YOLO

# --------- user params ----------
CAM_ID = 0
BACKGROUND_MARKER_IDS = [10,11,12,13]  # paste multiple markers around board
PLANE_MARKER_LENGTH = 0.064
CUBE_IDS = [0,1,2,3]
CUBE_MARKER_LENGTH = 0.0315
TIP_OFFSET = np.array([0.0, -0.020, 0.0])  # m from cube to tip
CONTACT_THRESH = 0.002
YOLO_MODEL = "best_kpt.pt"  # model that can output solder tip keypoint (optional)
LOG_PATH = "compare_contacts.csv"
# ---------------------------------

def load_camera_calibration(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened(): raise FileNotFoundError("calibration.yaml missing")
    K = fs.getNode("camera_matrix").mat(); dist = fs.getNode("dist_coeff").mat(); fs.release()
    return K, dist

def rvec_tvec_to_T(rvec, tvec):
    R,_ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.reshape(3,); return T
def invT(T): R=T[:3,:3]; t=T[:3,3]; Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti
def fit_plane(points):
    centroid = points.mean(axis=0)
    _,_,Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return centroid, normal

# main
def main():
    K, dist = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(CAM_ID)
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # load yolo model for tip estimation (optional)
    try:
        yolo = YOLO(YOLO_MODEL)
        yolo_avail = True
    except Exception as e:
        print("YOLO load failed or model not provided:", e); yolo_avail=False

    # CSV header
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["time","frame","cube_tip_z_m","solder_tip_z_m","cube_solder_dist_m","cube_plane_contact","solder_plane_contact","cube_solder_contact"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, dict_aruco)

        # collect 3D corners of background markers to fit plane
        marker3d_pts = []
        if ids is not None:
            for i, idv in enumerate(ids.flatten()):
                # estimate each marker pose and collect its 4 corners (in camera coords)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[i]], PLANE_MARKER_LENGTH, K, dist)
                rvec, tvec = rvecs[0].reshape(3,), tvecs[0].reshape(3,)
                R,_ = cv2.Rodrigues(rvec.reshape(3,1))
                s = PLANE_MARKER_LENGTH / 2.0
                obj = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)
                pts3 = (R @ obj.T).T + tvec.reshape(1,3)
                marker3d_pts.append(pts3)

        plane_ok = False
        if len(marker3d_pts) > 0:
            all_pts = np.vstack(marker3d_pts)
            plane_point, plane_normal = fit_plane(all_pts)
            plane_ok = True
            # draw centroid
            p2d, _ = cv2.projectPoints(np.array([plane_point]), np.zeros(3), np.zeros(3), K, dist)
            cv2.circle(frame, tuple(p2d.ravel().astype(int)), 4, (255,0,0), -1)

        # cube board pose
        cube_board = aruco.Board_create(
            objPoints=[np.array([[-CUBE_MARKER_LENGTH/2, CUBE_MARKER_LENGTH/2,0],
                                 [CUBE_MARKER_LENGTH/2, CUBE_MARKER_LENGTH/2,0],
                                 [CUBE_MARKER_LENGTH/2,-CUBE_MARKER_LENGTH/2,0],
                                 [-CUBE_MARKER_LENGTH/2,-CUBE_MARKER_LENGTH/2,0]], dtype=np.float32) for _ in range(4)],
            dictionary=dict_aruco, ids=np.array([[CUBE_IDS[0]],[CUBE_IDS[1]],[CUBE_IDS[2]],[CUBE_IDS[3]]], dtype=np.int32))
        cube_ok=False; solder_ok=False

        if ids is not None:
            retval, rvec_b, tvec_b = aruco.estimatePoseBoard(corners, ids, cube_board, K, dist, None, None)
            if retval > 0:
                T_cube = rvec_tvec_to_T(rvec_b.reshape(3,), tvec_b.reshape(3,))
                cube_ok=True
                cv2.drawFrameAxes(frame, K, dist, rvec_b, tvec_b, 0.03)
            # try solder marker id=8 if exists
            if 8 in ids.flatten():
                idx = list(ids.flatten()).index(8)
                r_s, t_s, _ = aruco.estimatePoseSingleMarkers([corners[idx]], 0.02, K, dist)
                T_solder = rvec_tvec_to_T(r_s[0].reshape(3,), t_s[0].reshape(3,))
                solder_ok=True
                cv2.drawFrameAxes(frame, K, dist, r_s, t_s, 0.02)

        # if no solder marker and YOLO available, try to get solder tip via YOLO
        yolo_tip_cam = None
        if not solder_ok and yolo_avail:
            res = yolo.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
            kp = None
            if hasattr(res, "keypoints") and res.keypoints is not None:
                kps = getattr(res.keypoints, "xy", None)
                if kps is not None and len(kps)>0:
                    kp = kps[0][0]  # adjust index for your model: this expects the first keypoint is solder tip
            if kp is None and hasattr(res, "boxes") and len(res.boxes)>0:
                box = res.boxes.xyxy[0]
                x1,y1,x2,y2 = map(float, box); kp = ((x1+x2)/2.0, (y1+y2)/2.0)
            if kp is not None and plane_ok:
                # project pixel to camera ray and intersect with plane to get 3D point
                u,v = float(kp[0]), float(kp[1])
                fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
                x = (u - cx) / fx; y = (v - cy) / fy
                ray = np.array([x,y,1.0]); ray /= np.linalg.norm(ray)
                origin = np.array([0.0,0.0,0.0])
                denom = plane_normal.dot(ray)
                if abs(denom) > 1e-6:
                    t = plane_normal.dot(plane_point - origin) / denom
                    if t > 0:
                        yolo_tip_cam = origin + ray * t
                        # draw
                        p2d,_ = cv2.projectPoints(np.array([yolo_tip_cam]), np.zeros(3), np.zeros(3), K, dist)
                        cv2.circle(frame, tuple(p2d.ravel().astype(int)), 5, (0,255,255), -1)
                        solder_ok = True  # treat as found
                        T_solder = np.eye(4); T_solder[:3,3] = yolo_tip_cam

        # compute distances
        cube_tip_z = float('nan'); solder_tip_z = float('nan'); cube_solder_d = float('nan')
        c_p = s_p = c_s = 0
        if plane_ok and cube_ok:
            cam_tip = (T_cube @ np.hstack([TIP_OFFSET.reshape(3,),1.0]))[:3]
            cube_tip_z = abs(np.dot(plane_normal, cam_tip - plane_point))
            s_p = int(cube_tip_z < CONTACT_THRESH)
            # draw cube tip projection
            p2d,_ = cv2.projectPoints(np.array([cam_tip]), np.zeros(3), np.zeros(3), K, dist)
            cv2.circle(frame, tuple(p2d.ravel().astype(int)), 5, (255,0,0), -1)
        if plane_ok and solder_ok:
            cam_solder = (T_solder @ np.array([0,0,0,1]))[:3]
            solder_tip_z = abs(np.dot(plane_normal, cam_solder - plane_point))
            s_s = int(solder_tip_z < CONTACT_THRESH)
            p2d,_ = cv2.projectPoints(np.array([cam_solder]), np.zeros(3), np.zeros(3), K, dist)
            cv2.circle(frame, tuple(p2d.ravel().astype(int)), 5, (0,255,0), -1)
        if cube_ok and solder_ok:
            cam_tip = (T_cube @ np.hstack([TIP_OFFSET.reshape(3,),1.0]))[:3]
            cam_solder = (T_solder @ np.array([0,0,0,1]))[:3]
            cube_solder_d = np.linalg.norm(cam_tip - cam_solder)
            c_s = int(cube_solder_d < CONTACT_THRESH)

        # write CSV
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([time.time(), frame_idx, cube_tip_z, solder_tip_z, cube_solder_d, s_p, s_s, c_s])

        # overlay
        cv2.putText(frame, f"cube-plane(mm): {cube_tip_z*1000 if not math.isnan(cube_tip_z) else float('nan'):.1f}", (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"solder-plane(mm): {solder_tip_z*1000 if not math.isnan(solder_tip_z) else float('nan'):.1f}", (8,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"cube-solder(mm): {cube_solder_d*1000 if not math.isnan(cube_solder_d) else float('nan'):.1f}", (8,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Aruco Multi + YOLO solder", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
