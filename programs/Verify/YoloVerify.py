from ultralytics import YOLO

# 学習済みのYOLOv11ポーズモデルをロード
# (例: 'yolov11-n-pose.pt' や独自に学習したモデル)
model = YOLO('best.pt')

# 評価の実行
# Ultralyticsは、`val`ディレクトリに関連付けられた
# COCO形式のアノテーションファイル（ステップ2で作成したもの）を探します。
# data引数には .yaml ファイルを指定するのが一般的です。
# もし .yaml がうまく機能しない場合、val() の引数を調整する必要があるかもしれません。

# Ultralytics v8+ スタイルの評価（YOLOv11がこれに従うと仮定）
# ステップ2で作成した gt.json を 'val' と同じ階層（または 'annotations' 内）に置く
# (例: .../FreiHAND_pub_v2/annotations/freihand_eval_coco.json)
# そして data.yaml でアノテーションパスを指定する必要があるかもしれません。

# --- UltralyticsがCOCO GT JSONを直接使う方法 ---
# Ultralytics v8+ では、val() 実行時にアノテーションを自動で探します。
# data.yaml の設定が重要です。
# data.yaml の 'path' と 'val' を正しく設定し、
# COCO JSONファイル (freihand_eval_coco.json) を
# .../annotations/ ディレクトリに配置するのが標準的です。

# data='freihand_eval.yaml' を指定して実行
results = model.val(data='../../dataset/freihand_eval.yaml', 
                    split='val',
                    imgsz=224,
                    batch=16)

# 結果の表示
print("--- YOLOv11 Evaluation Results (mAP OKS) ---")
print(results.pose) # mAP50-95(P), mAP50(P) などが表示されます
