from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

results = model.train(
    data="data_keypoint.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="solder_tip_kpt",
    device='cpu'
)
