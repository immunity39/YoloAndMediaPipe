from ultralytics import YOLO
import torch, gc, os

# 強制同期でエラー位置を明示化
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# CuDNN関連キャッシュを無効化
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# メモリ整理
gc.collect()
torch.cuda.empty_cache()


def main():
#     # Load a model
    model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

#     # Train the model
#     results = model.train(
#         data="hand-keypoints.yaml",
#         epochs=100,
#         imgsz=640,
#         batch=16,
#         patience=20,
#         workers=8,
#         name="yolo11n-hand-pose",
#         device='cuda:0'
#     )

    results = model.train(
        data="hand-keypoints.yaml",
        epochs=150,        # ← エポック延長（100→150）
        imgsz=640,
        batch=16,
        patience=30,       # ← 早期終了を少し緩める
        augment=True,      # ← 明示的に有効化
        mosaic=1.0,        # ← モザイク合成の強化
        mixup=0.2,         # ← mixupデータ生成を追加
        degrees=15,        # ← 回転角度許容範囲を拡大
        translate=0.1,     # ← 平行移動を導入
        scale=0.5,         # ← スケール拡張
        shear=2.0,         # ← 軽いせん断変形
        device=0,
        workers=8,
    )

if __name__ == "__main__":
    main()
