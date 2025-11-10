import mediapipe as mp
import cv2
import json
import os
from tqdm import tqdm

# --- MediaPipeセットアップ ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1, # FreiHANDは通常片手
    min_detection_confidence=0.1 # 感度を上げる
)

# --- FreiHAND評価画像ディレクトリ ---
image_dir = '/path/to/FreiHAND_pub_v2/evaluation/rgb/'
# ステップ2で作成した gt.json をロード (image_id を参照するため)
gt_json_path = '/path/to/FreiHAND_pub_v2/freihand_eval_coco.json'
with open(gt_json_path, 'r') as f:
    gt_data = json.load(f)

# --- COCO結果フォーマット用のリスト ---
coco_results = []
score_threshold = 0.5 # MediaPipeの検出信頼度の閾値

print("Running MediaPipe inference on FreiHAND evaluation set...")

# gt_data の images リストをループ
for image_info in tqdm(gt_data['images']):
    image_id = image_info['id']
    file_name = image_info['file_name']
    image_path = os.path.join(image_dir, file_name)
    
    image = cv2.imread(image_path)
    if image is None:
        continue
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # 検出された手（通常1つ）
        hand_landmarks = results.multi_hand_landmarks[0]
        # 検出スコア (Handedness score)
        score = results.multi_handedness[0].classification[0].score
        
        if score < score_threshold:
            continue

        keypoints_coco = []
        valid_points = []
        
        # 21キーポイントを処理
        for i in range(21):
            lm = hand_landmarks.landmark[i]
            # MediaPipeの座標は [0.0, 1.0] の正規化座標
            # COCO形式はピクセル座標 [x, y, visibility]
            x_px = lm.x * image_info['width']
            y_px = lm.y * image_info['height']
            
            # MediaPipeのvisibility (v) は [0.0, 1.0] の信頼度
            # COCOの v は 0(なし), 1(隠れ), 2(見える)
            # ここでは単純化し、lm.visibility > 閾値なら v=2 とする
            visibility_score = lm.visibility if hasattr(lm, 'visibility') else 1.0
            v = 2 # if visibility_score > 0.1 else 0 # 閾値は要調整
            
            keypoints_coco.extend([round(x_px, 2), round(y_px, 2), v])
            if v > 0:
                valid_points.append((x_px, y_px))

        if not valid_points:
            continue
            
        # BBox計算 (推論結果から)
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]
        xmin = max(0, min(x_coords))
        ymin = max(0, min(y_coords))
        xmax = min(image_info['width'], max(x_coords))
        ymax = min(image_info['height'], max(y_coords))
        
        # COCO結果フォーマット:
        # { "image_id": int, 
        #   "category_id": 1, 
        #   "keypoints": [x1,y1,v1,...], 
        #   "score": float }
        coco_results.append({
            "image_id": image_id,
            "category_id": 1,
            "keypoints": keypoints_coco,
            "score": float(score) # 手全体の検出スコア
        })

hands.close()

# --- 結果 (results.json) を保存 ---
output_results_path = '/path/to/mediapipe_results.json'
with open(output_results_path, 'w') as f:
    json.dump(coco_results, f, indent=4)

print(f"MediaPipe COCO results file saved to: {output_results_path}")
