import json
import numpy as np
import os
from PIL import Image

# --- FreiHANDデータセットへのパス ---
base_dir = r'..\..\dataset'
image_dir = os.path.join(base_dir, 'evaluation', 'rgb')
anno_xyz_path = os.path.join(base_dir, 'evaluation_xyz.json')
anno_k_path = os.path.join(base_dir, 'evaluation_K.json')
output_json_path = os.path.join(base_dir, 'freihand_eval_coco.json') # これが gt.json

# --- FreiHANDのアノテーションをロード ---
with open(anno_xyz_path, 'r') as f:
    xyz_list = json.load(f) # 3Dキーポイント (21, 3)
with open(anno_k_path, 'r') as f:
    k_list = json.load(f)   # カメラ内部パラメータ (3, 3)

coco_output = {
    "info": {"description": "FreiHAND Evaluation Set in COCO format"},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "hand",
        "supercategory": "hand",
        # FreiHAND/MediaPipeのキーポイント順序を定義（後述）
        "keypoints": [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ],
        "skeleton": [] # 必要なら定義
    }]
}

# --- 画像とアノテーションを処理 ---
image_id_counter = 0
annotation_id_counter = 0

# FreiHANDの評価画像は通常 00000000.jpg から 00003959.jpg
num_images = len(xyz_list) 

for i in range(num_images):
    image_id_counter += 1
    image_file = f'{i:08d}.jpg'
    image_path = os.path.join(image_dir, image_file)
    
    # 画像サイズを取得 (FreiHANDは通常 224x224)
    # try:
    #     with Image.open(image_path) as img:
    #         width, height = img.size
    # except FileNotFoundError:
    #     print(f"Warning: Image not found {image_path}")
    #     continue
    width, height = 224, 224 # 固定サイズと仮定

    # --- images セクション ---
    coco_output["images"].append({
        "id": image_id_counter,
        "file_name": image_file,
        "width": width,
        "height": height,
        "license": None, "coco_url": "", "date_captured": "", "flickr_url": ""
    })

    # --- 3D -> 2D 座標変換 ---
    xyz = np.array(xyz_list[i]) # (21, 3)
    K = np.array(k_list[i])     # (3, 3)
    
    # プロジェクション: (u, v, z) = K @ xyz.T
    # (x, y) = (u/z, v/z)
    uvz = K @ xyz.T
    uvz = uvz.T # (21, 3)
    
    # zが0または非常に小さい場合の除算エラーを回避
    # zが0より大きい場合のみ2D座標を計算
    keypoints_2d = []
    valid_points = []
    
    for j in range(21):
        z = uvz[j, 2]
        if z > 1e-6: # zが正の場合のみ有効
            x = uvz[j, 0] / z
            y = uvz[j, 1] / z
            # COCO形式: [x, y, visibility]
            # FreiHANDはすべて見える前提 (v=2) かもしれないが、画像外チェックも必要
            v = 2 # 2: 見えておりアノテーションあり
            if not (0 <= x < width and 0 <= y < height):
                v = 0 # 0: アノテーションなし (画像外)
                
            keypoints_2d.extend([round(x, 2), round(y, 2), v])
            if v > 0:
                valid_points.append((x, y))
        else:
            keypoints_2d.extend([0.0, 0.0, 0]) # v=0: アノテーションなし

    if not valid_points:
        # この画像には有効なキーポイントが一つもない
        continue 
        
    # --- BBoxの計算 ---
    # 有効なキーポイントからバウンディングボックスを作成
    x_coords = [p[0] for p in valid_points]
    y_coords = [p[1] for p in valid_points]
    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)
    
    # 少しパディング（マージン）を追加
    padding = 10 
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(width, xmax + padding)
    ymax = min(height, ymax + padding)

    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    
    # COCO形式: [xmin, ymin, width, height]
    bbox = [round(xmin, 2), round(ymin, 2), round(bbox_w, 2), round(bbox_h, 2)]
    area = bbox_w * bbox_h
    
    # --- annotations セクション ---
    annotation_id_counter += 1
    coco_output["annotations"].append({
        "id": annotation_id_counter,
        "image_id": image_id_counter,
        "category_id": 1,
        "keypoints": keypoints_2d,
        "num_keypoints": sum(1 for v in keypoints_2d[2::3] if v > 0),
        "bbox": bbox,
        "area": round(area, 2),
        "iscrowd": 0
    })

# --- COCO形式のJSONファイルとして保存 ---
with open(output_json_path, 'w') as f:
    json.dump(coco_output, f, indent=4)

print(f"COCO format GT file saved to: {output_json_path}")
