import cv2
import os

OUT_DIR = "images"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
i = 0
print("Press 'q' to stop, 's' to save a frame")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("cam", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        fname = f"img_{i:05d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), frame)
        print("Saved", fname)
        i += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
