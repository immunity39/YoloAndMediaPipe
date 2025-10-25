import cv2, numpy as np, mediapipe as mp, time, math
from cv2 import aruco

# ========== parameters ==========
CAM_ID = 0
BOARD_MARKER_IDS = [4,5,6,7]   # top-left, top-right, bottom-left, bottom-right
BOARD_MARKER_LENGTH = 0.051    # 5.1cm
CONTACT_THRESHOLD = 0.002      # 2mm
# =================================

def load_camera_calibration(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def build_board_object_points(marker_length):
    s = marker_length/2
    objPoints = [
        np.array([[-s, s, 0], [ s, s, 0], [ s,-s, 0], [-s,-s, 0]], dtype=np.float32),
        np.array([[+0.101, 0, 0], [+0.101+marker_length, 0, 0], [+0.101+marker_length, -marker_length, 0], [+0.101, -marker_length, 0]], dtype=np.float32)
    ]  # 修正例：マーカ間隔は必要に応じ変更
    return objPoints

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.reshape(3,)
    return T

def invT(T):
    R=T[:3,:3]; t=T[:3,3]
    Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t
    return Ti

def pixel_to_cam_ray(u, v, K):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    v = np.array([x, y, 1.0])
    return v / np.linalg.norm(v)

def ray_plane_intersection(origin, dir_vec, plane_point, plane_normal):
    denom = plane_normal.dot(dir_vec)
    if abs(denom) < 1e-6: return None
    t = plane_normal.dot(plane_point - origin) / denom
    if t < 0: return None
    return origin + t*dir_vec

def main():
    K, dist = load_camera_calibration()
    cap = cv2.VideoCapture(CAM_ID)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    ar_params = aruco.DetectorParameters()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ar_dict)

        plane_ok=False
        if ids is not None:
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, 
                aruco.GridBoard_create(2,2,0.051,0.01,ar_dict), K, dist)
            if retval>0:
                plane_ok=True
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)
                T = rvec_tvec_to_T(rvec, tvec)
                plane_point = T[:3,3]
                plane_normal = T[:3,2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                idx_tip = handLms.landmark[8]  # index finger tip
                h, w, _ = frame.shape
                px, py = int(idx_tip.x * w), int(idx_tip.y * h)
                cv2.circle(frame, (px, py), 5, (0,255,0), -1)

                if plane_ok:
                    ray = pixel_to_cam_ray(px, py, K)
                    origin = np.array([0,0,0],dtype=float)
                    p_int = ray_plane_intersection(origin, ray, plane_point, plane_normal)
                    if p_int is not None:
                        dist_m = abs(np.dot(plane_normal, p_int - plane_point))
                        color = (0,0,255) if dist_m<CONTACT_THRESHOLD else (0,255,0)
                        cv2.putText(frame, f"{dist_m*1000:.1f}mm", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.circle(frame, (px,py), 6, color, -1)

        cv2.imshow("MediaPipe + ArUco Contact", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
