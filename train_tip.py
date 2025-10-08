from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # 小型モデルをベースに

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="solder_tip",
    device=0  # GPUがない場合は 'cpu'
)
