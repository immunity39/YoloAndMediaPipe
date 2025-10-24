from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="hand-keypoints.yaml",
        epochs=30,
        imgsz=640,
        name="yolo11n-hand-pose",
        device='cuda:0'
    )

if __name__ == "__main__":
    main()