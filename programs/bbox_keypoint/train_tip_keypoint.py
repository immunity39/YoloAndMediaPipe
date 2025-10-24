from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-pose.pt")

    results = model.train(
        data="data_keypoint.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="solder_tip_kpt",
        device='cpu'
    )

if __name__ == "__main__":
    main()