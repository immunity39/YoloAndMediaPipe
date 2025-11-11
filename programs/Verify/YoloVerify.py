from ultralytics import YOLO

# 学習済みのYOLOv11ポーズモデルをロード
# (例: 'yolov11-n-pose.pt' や独自に学習したモデル)
model = YOLO('best.pt')

results = model.val(data='data.yaml', 
                    split='val',
                    imgsz=224,
                    batch=16)

# 結果の表示
print("--- YOLOv11 Evaluation Results (mAP OKS) ---")
print(results.pose) # mAP50-95(P), mAP50(P) などが表示されます
