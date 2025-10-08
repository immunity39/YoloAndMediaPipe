import cv2
import numpy as np
from ultralytics import YOLO
import time
import yaml
import math

# load camera calibration
def load_calib(path="calibration.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

MODEL_PATH = "best_tip_model.pt"  # fine-tuned model path (学習済みがなければカテゴリ 'solder_tip' 等で学習)
CONF_THRESH = 0.25

def main():
    K, dist = load_calib("calibration.yaml")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # inference
        results = model.predict(source=frame, conf=CONF_THRESH, imgsz=640, device='0', verbose=False)
        # results is a list; take first
        r = results[0]
        # r.boxes.xyxy  r.boxes.conf  r.boxes.cls  (for yolov8)
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # compute center pixel as tip proxy
                cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame, f"{conf:.2f}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                # optional: project to camera coordinates -> need depth (z). Without depth, only bearing.
                # If you have stereo or known plane intersection, compute 3D.
        cv2.imshow("YOLO Tip", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
