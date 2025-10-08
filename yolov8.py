from ultralytics import YOLO
import cv2
import numpy as np
import time

# モデル読み込み（キーポイント対応モデルを学習している想定）
model = YOLO("yolov8n-pose.pt")  # 例: キーポイント/pose対応モデル

# カメラキャリブレーション（fx,fy,cx,cy）を事前に取得しておく
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
dist_coeffs = np.zeros(5)  # 実際にはcalibrateCameraで取得

# ArUco準備（基板上に置いたマーカーの3D座標を事前に定義）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# 基板上のマーカーのワールド座標（例: マーカーの中心の3D座標、mm単位）
# 例: marker_world = {id: (x_mm, y_mm, z_mm)}
marker_world = {0: (0.0, 0.0, 0.0), 1: (50.0, 0.0, 0.0), 2: (50.0, 40.0, 0.0), 3: (0.0, 40.0, 0.0)}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    # 1) ArUco検出 -> 基板平面のpose（rvec, tvec）
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    board_rvec = None
    board_tvec = None
    if ids is not None:
        # 対応点作成
        image_points = []
        object_points = []
        for i, cid in enumerate(ids.flatten()):
            if int(cid) in marker_world:
                # use marker center -> ここはマーカー四隅の座標でより正確にすること
                c = corners[i].reshape(4,2).mean(axis=0)
                image_points.append(c)
                object_points.append(marker_world[int(cid)])
        if len(image_points) >= 4:
            image_points = np.array(image_points, dtype=np.float32)
            object_points = np.array(object_points, dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if ok:
                board_rvec, board_tvec = rvec, tvec

    # 2) YOLO 推論（キーポイント or bbox）
    results = model(frame)[0]  # シンプル扱い
    tip_xy = None
    for det in results.boxes:
        cls = int(det.cls[0])
        # ここではクラス番号でtipを判定する想定
        if cls == TIP_CLASS_ID:
            x1,y1,x2,y2 = map(int, det.xyxy[0])
            # キーポイントがあるなら det.keypoints を使う
            if hasattr(det, "keypoints") and det.keypoints is not None:
                tip_xy = det.keypoints[0]  # (x,y,confidence)
            else:
                tip_xy = ((x1+x2)/2, (y1+y2)/2, 1.0)

    # 3) 2D->3D (solvePnP for tip) : 逆投影（単一点では不定だが、基板平面が既知なら平面との交点を取る）
    contact = False
    if tip_xy is not None and board_rvec is not None:
        # カメラ座標系における基板平面の方程式を計算する
        R, _ = cv2.Rodrigues(board_rvec)
        plane_normal = R[:,2]  # 3rd column がZ軸（設定による）
        board_origin = board_tvec.reshape(3)
        # 画素 u,v をカメラ座標系の視線ベクトルに
        u, v = tip_xy[0], tip_xy[1]
        x_cam = np.linalg.inv(camera_matrix) @ np.array([u, v, 1.0])
        # 視線パラメータ t: origin + s * dir intersects plane -> solve for s
        cam_pos = np.zeros(3)  # カメラ原点
        dir_vec = x_cam / np.linalg.norm(x_cam)
        denom = plane_normal.dot(dir_vec)
        if abs(denom) > 1e-6:
            s = plane_normal.dot(board_origin - cam_pos) / denom
            tip_3d = s * dir_vec  # in same units as board_origin (mm if tvec in mm)
            # Z距離差
            z_diff = plane_normal.dot(tip_3d - board_origin)
            # 閾値（mm単位）
            if z_diff <= 1.5:  # 例: 1.5mm
                contact = True

    # 可視化
    label = "CONTACT" if contact else "NO"
    cv2.putText(frame, label, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if not contact else (0,0,255), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
