import cv2
import numpy as np
import glob
import os

# --- 設定 ---
CHECKERBOARD = (6, 9)  # 列×行（交点数）
square_size = 0.018    # 1マスのサイズ [m] (例: 18mm)

# --- 3D点の座標系を生成 ---
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# --- 格納リスト ---
objpoints = []  # 3D点
imgpoints = []  # 2D点

# --- キャリブレーション画像読み込み ---
images = glob.glob('calib_images/*.jpg')
if len(images) == 0:
    print("❌ エラー: 'calib_images' フォルダに .jpg 画像がありません。")
    exit()

print(f"🔍 検出対象画像: {len(images)} 枚")

# --- チェッカーボード検出 ---
for fname in images:
    print(f"▶ {fname}")
    img = cv2.imread(fname)
    if img is None:
        print(f"⚠ 読み込み失敗: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        # サブピクセル精度で補正
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Chessboard', img)
        key = cv2.waitKey(200)
        if key == 27:  # ESCで中断
            break
    else:
        print(f"⚠ コーナー未検出: {fname}")

cv2.destroyAllWindows()

if len(objpoints) == 0:
    print("❌ 有効なチェッカーボードが検出されませんでした。")
    exit()

# --- カメラキャリブレーション ---
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# --- 結果出力 ---
print("\n✅ キャリブレーション完了")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# --- 保存 ---
os.makedirs("output", exist_ok=True)
save_path = "output/calibration.yaml"
fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeff", dist_coeffs)
fs.release()
print(f"\n💾 保存完了: {save_path}")
