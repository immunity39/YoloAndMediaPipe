from ultralytics import YOLO

# 学習済みのYOLOポーズモデルをロード
YOLO_MODEL = "runs/pose/train/weights/best.pt"
model = YOLO(YOLO_MODEL)

print("--- Running YOLOv11 Validation with low confidence threshold ---")

# ★★★ conf=0.01 を追加 ★★★
# 信頼度 1% 以上の検出をすべて評価対象とする
results = model.val(data='data.yaml', 
                    split='val',
                    imgsz=224,
                    batch=16,
                    conf=0.01)  # <-- この引数を追加

# --- mAP (OKS) の取得 ---
map_50_95 = results.pose.map
map_50 = results.pose.map50
map_75 = results.pose.map75

print("\n--- YOLOv11 Evaluation Results (mAP OKS) ---")
print(f"mAP(P) @ .50:.95 = {map_50_95}")
print(f"mAP(P) @ .50      = {map_50}")
print(f"mAP(P) @ .75      = {map_75}")

# --- mAP (BBox) の取得 ---
bbox_map_50_95 = results.box.map
bbox_map_50 = results.box.map50

print("\n--- YOLOv11 Evaluation Results (mAP BBox) ---")
print(f"mAP(B) @ .50:.95 = {bbox_map_50_95}")
print(f"mAP(B) @ .50      = {bbox_map_50}")

# (ファイル保存処理は省略)
