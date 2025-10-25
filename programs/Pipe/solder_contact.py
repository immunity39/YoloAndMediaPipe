import cv2, numpy as np, mediapipe as mp, time, math
from cv2 import aruco

# === Parameters ===
CAM_ID = 0
PLANE_IDS = [4,5,6,7]      # 基板ボード 4隅
CUBE_IDS = [0,1,2,3]       # コテのキューブマーカ
MARKER_LEN_PLANE = 0.051
MARKER_LEN_CUBE  = 0.0315
CUBE_W, CUBE_D = 0.047, 0.040
TIP_OFFSET = np.array([0.0, -0.020, 0.00])
CONTACT_TH = 0.002  # m
# ===================

def load_camera_calibration(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def rvec_tvec_to_T(rvec, tvec):
    R,_=cv2.Rodrigues(rvec); T=np.eye(4);T[:3,:3]=R;T[:3,3]=tvec.reshape(3,)
    return T

def invT(T):
    R=T[:3,:3];t=T[:3,3];Ti=np.eye(4);Ti[:3,:3]=R.T;Ti[:3,3]=-R.T@t;return Ti

def build_cube(marker_len,w,d):
    X=np.array([1,0,0]);Y=np.array([0,1,0]);Z=np.array([0,0,1])
    hw=w/2;hd=d/2
    centers=[np.array([0,0,hd]), np.array([hw,0,0]), np.array([0,0,-hd]), np.array([-hw,0,0])]
    uvs=[(X,Y),(-Z,Y),(-X,Y),(Z,Y)]
    obj=[]
    for c,(u,v) in zip(centers,uvs):
        half=marker_len/2
        tl=c - u*half + v*half
        tr=c + u*half + v*half
        br=c + u*half - v*half
        bl=c - u*half - v*half
        obj.append(np.vstack([tl,tr,br,bl]).astype(np.float32))
    return obj

def main():
    K, dist = load_camera_calibration()
    mp_hands=mp.solutions.hands
    hands=mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    ar_dict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    cube_board=aruco.Board(objPoints=build_cube(MARKER_LEN_CUBE,CUBE_W,CUBE_D),
                            ids=np.array([[i] for i in CUBE_IDS],dtype=np.int32),
                            dictionary=ar_dict)
    cap=cv2.VideoCapture(CAM_ID)
    print("MediaPipe + ArUco (Cube + Plane + Hand) Contact Detection")

    while True:
        ret,frame=cap.read()
        if not ret: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners,ids,_=aruco.detectMarkers(gray,ar_dict)
        plane_ok=False; cube_ok=False
        if ids is not None:
            retval_p,rvec_p,tvec_p=aruco.estimatePoseBoard(corners,ids,
                aruco.GridBoard_create(2,2,MARKER_LEN_PLANE,0.01,ar_dict),
                K,dist,None,None)
            if retval_p>0: plane_ok=True; T_plane=rvec_tvec_to_T(rvec_p,tvec_p)
            retval_c,rvec_c,tvec_c=aruco.estimatePoseBoard(corners,ids,cube_board,K,dist,None,None)
            if retval_c>0: cube_ok=True; T_cube=rvec_tvec_to_T(rvec_c,tvec_c)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(rgb)
        if plane_ok and cube_ok and results.multi_hand_landmarks:
            plane_n=T_plane[:3,2];plane_p=T_plane[:3,3]
            # コテ先（キューブ）
            tip_cam=(T_cube @ np.hstack([TIP_OFFSET,1]))[:3]
            dist_tool=abs(np.dot(plane_n, tip_cam - plane_p))
            # 手先
            hand=results.multi_hand_landmarks[0]
            h,w,_=frame.shape
            px,py=int(hand.landmark[8].x*w),int(hand.landmark[8].y*h)
            cv2.circle(frame,(px,py),5,(0,255,0),-1)
            ray=np.array([(px-K[0,2])/K[0,0],(py-K[1,2])/K[1,1],1.0])
            ray/=np.linalg.norm(ray)
            p_int=None
            denom=plane_n.dot(ray)
            if abs(denom)>1e-6:
                t=plane_n.dot(plane_p)/denom
                p_int=ray*t
            dist_hand=abs(np.dot(plane_n,p_int-plane_p)) if p_int is not None else np.nan

            contact_tool=dist_tool<CONTACT_TH
            contact_hand=dist_hand<CONTACT_TH
            cv2.putText(frame,f"Tool:{dist_tool*1000:.1f}mm", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255) if contact_tool else (0,255,0),2)
            cv2.putText(frame,f"Hand:{dist_hand*1000:.1f}mm", (10,55), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255) if contact_hand else (0,255,0),2)

        cv2.imshow("MediaPipe Cube+Board Contact",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):break
    cap.release();cv2.destroyAllWindows()

if __name__=="__main__":
    main()
