from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/solder_tip/weights/best.pt")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, conf=0.4, imgsz=640, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Tip detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
