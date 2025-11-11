from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

# --- ステップ2で作成した「正解」アノテーション (gt.json) ---
gt_json_path = '/path/to/FreiHAND_pub_v2/freihand_eval_coco.json'
gt_json_path = r''

# --- ステップ4で作成した MediaPipe の「推論結果」 (results.json) ---
dt_json_path = '/path/to/mediapipe_results.json'
dt_json_path = r''

# --- COCO APIのロード ---
coco_gt = COCO(gt_json_path)       # 正解データをロード
coco_dt = coco_gt.loadRes(dt_json_path) # 推論結果をロード

# --- 評価オブジェクトの作成 (キーポイント評価) ---
coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints') # 'keypoints' を指定

# --- 評価の実行 ---
print("\n--- Running COCOeval for MediaPipe (mAP OKS) ---")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize() # mAP (OKS) が表示されます

# --- MediaPipeのmAP結果を取得 ---
mediapipe_map_50_95 = coco_eval.stats[0]
mediapipe_map_50 = coco_eval.stats[1]

#print(f"MediaPipe mAP (OKS) @ .50:.95 = {mediapipe_map_50_95}")
#print(f"MediaPipe mAP (OKS) @ .50 = {mediapipe_map_50}")


# --- 比較 ---
# 学習済みのYOLOv11ポーズモデルをロード
# 上記の MediaPipe の mAP 値を比較します。
model = YOLO('best.pt')

results = model.val(data='data.yaml', 
                    split='val',
                    imgsz=224,
                    batch=16)

# 結果の表示
print(f"\n--- Comparison ---")
print(f"YOLOv11 mAP@.50:.95: {results.pose.map}")
print(f"MediaPipe mAP@.50:.95: {mediapipe_map_50_95}")
print(f"\nYOLOv11 mAP@.50: {results.pose.map50}")
print(f"MediaPipe mAP@.50: {mediapipe_map_50}")
