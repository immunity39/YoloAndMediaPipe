import json
import numpy as np
import os
from PIL import Image

# --- FreiHANDデータセットへのパス ---
base_dir = '.'
image_dir = os.path.join(base_dir, 'evaluation', 'rgb')
anno_xyz_path = os.path.join(base_dir, 'evaluation_xyz.json')
anno_k_path = os.path.join(base_dir, 'evaluation_K.json')

output_json_path = os.path.join(base_dir, 'evaluation', 'annotations', 'person_keypoints_val.json')
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)


# --- FreiHANDのアノテーションをロード ---
with open(anno_xyz_path, 'r') as f:
    xyz_list = json.load(f)
with open(anno_k_path, 'r') as f:
    k_list = json.load(f)

# --- ★★★ 21個のシグマ値 ★★★ ---
hand_sigmas_21 = [
    0.035, 0.036, 0.036, 0.036, 0.036, 0.072, 0.072, 0.072, 0.072, 
    0.062, 0.062, 0.062, 0.062, 0.087, 0.087, 0.087, 0.087, 
    0.089, 0.089, 0.089, 0.089
]

coco_output = {
    "info": {"description": "FreiHAND Evaluation Set (ID 0) in COCO format"},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 0,
        "name": "hand",
        "supercategory": "hand",
        "keypoints": [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ],
        "skeleton": [],
        "sigmas": hand_sigmas_21
    }]
}

# --- 画像とアノテーションを処理 ---
image_id_counter = 0
annotation_id_counter = 0

num_images = len(xyz_list) 

for i in range(num_images):
    image_id_counter += 1
    image_file = f'{i:08d}.jpg'
    image_path = os.path.join(image_dir, image_file)
    
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
    xyz = np.array(xyz_list[i]) 
    K = np.array(k_list[i])     
    uvz = K @ xyz.T
    uvz = uvz.T 
    
    keypoints_2d = []
    valid_points = []
    
    for j in range(21):
        z = uvz[j, 2]
        if z > 1e-6:
            x = uvz[j, 0] / z
            y = uvz[j, 1] / z
            v = 2
            if not (0 <= x < width and 0 <= y < height):
                v = 0
            keypoints_2d.extend([round(x, 2), round(y, 2), v])
            if v > 0:
                valid_points.append((x, y))
        else:
            keypoints_2d.extend([0.0, 0.0, 0])

    if not valid_points:
        continue 
        
    # --- BBoxの計算 ---
    x_coords = [p[0] for p in valid_points]
    y_coords = [p[1] for p in valid_points]
    xmin = max(0, min(x_coords) - 10) # Padding
    ymin = max(0, min(y_coords) - 10) # Padding
    xmax = min(width, max(x_coords) + 10) # Padding
    ymax = min(height, max(y_coords) + 10) # Padding
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    bbox = [round(xmin, 2), round(ymin, 2), round(bbox_w, 2), round(bbox_h, 2)]
    area = bbox_w * bbox_h
    
    # --- annotations セクション ---
    annotation_id_counter += 1
    coco_output["annotations"].append({
        "id": annotation_id_counter,
        "image_id": image_id_counter,
        "category_id": 0,
        "keypoints": keypoints_2d,
        "num_keypoints": sum(1 for v in keypoints_2d[2::3] if v > 0),
        "bbox": bbox,
        "area": round(area, 2),
        "iscrowd": 0
    })

# --- COCO形式のJSONファイルとして保存 ---
with open(output_json_path, 'w') as f:
    json.dump(coco_output, f)

print(f"COCO format GT file (with category_id=0) saved to: {output_json_path}")
