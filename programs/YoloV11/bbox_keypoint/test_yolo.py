from ultralytics import YOLO

dataset_path = "datasets"

model = YOLO("yolov8n-cls.pt")

results = model.train(data=dataset_path, epochs=100, device="cuda")
