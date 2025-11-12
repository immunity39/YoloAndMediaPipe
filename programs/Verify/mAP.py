from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import numpy as np

gt_json_path = r'evaluation/annotations/keypoint_val.json'
dt_json_path = 'mediapipe_results.json'

# --- COCO APIのロード ---
coco_gt = COCO(gt_json_path)       # 正解データをロード
coco_dt = coco_gt.loadRes(dt_json_path) # 推論結果をロード

# --- 評価オブジェクトの作成 (キーポイント評価) ---
coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints') # 'keypoints' を指定

hand_sigmas_21 = np.array([
    0.035, 0.036, 0.036, 0.036, 0.036, 0.072, 0.072, 0.072, 0.072, 
    0.062, 0.062, 0.062, 0.062, 0.087, 0.087, 0.087, 0.087, 
    0.089, 0.089, 0.089, 0.089
])
coco_eval.params.kpt_oks_sigmas = hand_sigmas_21

# --- 評価の実行 ---
print("\n--- Running COCOeval for MediaPipe (mAP OKS) ---")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize() # mAP (OKS) が表示されます

# --- MediaPipeのmAP結果を取得 ---
mediapipe_map_50_95 = coco_eval.stats[0]
mediapipe_map_50 = coco_eval.stats[1]

print(f"MediaPipe mAP (OKS) @ .50:.95 = {mediapipe_map_50_95}")
print(f"MediaPipe mAP (OKS) @ .50 = {mediapipe_map_50}")

Path = 'result'
filename = 'mediapipe_map_results.txt'
with open(f'{Path}/{filename}', 'w') as f:
    f.write("--- MediaPipe Evaluation Results (mAP OKS) ---\n")
    f.write(f"mAP (OKS) @ .50:.95 = {mediapipe_map_50_95}\n")
    f.write(f"mAP (OKS) @ .50      = {mediapipe_map_50}\n")

# --- 比較 ---
# Yolo11の評価結果を取得
# YOLO_RESULTS = r'yolo_pose_results.txt'
# results = YOLO_RESULTS
# # 上記の MediaPipe の mAP 値を比較します。

# # 結果の表示
# print(f"\n--- Comparison ---")
# print(f"YOLOv11 mAP@.50:.95: {results.pose.map}")
# print(f"MediaPipe mAP@.50:.95: {mediapipe_map_50_95}")
# print(f"\nYOLOv11 mAP@.50: {results.pose.map50}")
# print(f"MediaPipe mAP@.50: {mediapipe_map_50}")
