from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data_bbox.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="solder_tip",
    device='cpu'
)
