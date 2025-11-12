from ultralytics import YOLO

# 学習済みのYOLOポーズモデルをロード
YOLO_MODEL = "runs/pose/train/weights/best.pt"
model = YOLO(YOLO_MODEL)

results = model.val(data='data.yaml', 
                    split='val',
                    imgsz=224,
                    batch=16)

# --- mAP (OKS) の正しい取得方法 ---

# 1. コンソールにサマリー（表）を表示する (推奨)
# (results.pose ではなく、results 自体を出力します)
print(results)

# 2. 特定のmAPスコアを数値として取得する
map_50_95 = results.pose.map    # mAP(P) @ .50:.95 (COCO標準)
map_50 = results.pose.map50     # mAP(P) @ .50
map_75 = results.pose.map75     # mAP(P) @ .75

print("\n--- YOLOv11 Evaluation Results (mAP OKS) ---")
print(f"mAP(P) @ .50:.95 = {map_50_95}")
print(f"mAP(P) @ .50      = {map_50}")
print(f"mAP(P) @ .75      = {map_75}")

# ファイルへの書き出し
output_results_path = './yolo11_results.txt'
with open(output_results_path, 'w') as f:
    f.write("--- YOLOv11 Evaluation Results (mAP OKS) ---\n")
    f.write(f"mAP(P) @ .50:.95 = {map_50_95}\n")
    f.write(f"mAP(P) @ .50      = {map_50}\n")
    f.write(f"mAP(P) @ .75      = {map_75}\n")
    
    # BBox (検出) の mAP も取得可能
    f.write("\n--- YOLOv11 Evaluation Results (mAP BBox) ---\n")
    f.write(f"mAP(B) @ .50:.95 = {results.box.map}\n")
    f.write(f"mAP(B) @ .50      = {results.box.map50}\n")

print(f"YOLOv11 mAP results saved to {output_results_path}")
