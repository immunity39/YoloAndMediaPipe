from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="hand-keypoints.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        workers=8,
        name="yolo11n-hand-pose",
        device='cuda:0'
    )

if __name__ == "__main__":
    main()