from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 学習済みモデルのパスを指定（例: 'best.pt'）
model_path = 'model/best_model/best.pt'      #モデルを変更してください
model = YOLO(model_path,task='detect')
# 推論する画像が入っているフォルダのパス
image_folder = 'data/pepper_data/image_1/color' #画像の入っているディレクトリに変更
# 出力アノテーションデータの保存先パス
output_folder = 'label/output/'                 #出力先のディレクトリに変更
os.makedirs(output_folder, exist_ok=True) 
output_image = 'label/output/label_image'

# 推論を実行
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  
        image_path = os.path.join(image_folder, image_file)
        results = model.predict(image_path, conf = 0.75) # confは閾値

        # 画像サイズを取得
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        # 各画像に対する推論結果を保存
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")

        with open(output_path, 'w') as f:
            for result, cls in zip(results[0].boxes.xywhn, results[0].boxes.cls):
            	print(result)
            	x_center, y_center, width, height = result.tolist()
            	
            	# 正規化された座標（0〜1）からピクセル座標に変換
                x_center = int(x_center * img_width)
                y_center = int(y_center * img_height)
                width = int(width * img_width)
                height = int(height * img_height)
                cls = int(cls.item())

                # 結果をtxtファイルに書き込む
                f.write(f"{int(cls)} {x_center / img_width:.6f} {y_center / img_height:.6f} {width / img_width:.6f} {height / img_height:.6f}\n")
                print(f"{int(cls)} {x_center / img_width:.6f} {y_center / img_height:.6f} {width / img_width:.6f} {height / img_height:.6f}")

                # バウンディングボックスの座標計算
                x1 = x_center - width // 2
                y1 = y_center - height // 2
                x2 = x_center + width // 2
                y2 = y_center + height // 2

                # 画像内で座標をクリップ（範囲外の場合）
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))

                # バウンディングボックスの描画
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        print(f"Saved annotations for {image_file} to {output_path}")
        
        #画像を保存
        annotated_img_path = os.path.join(output_image, f"{os.path.splitext(image_file)[0]}_annotated.jpg")
        cv2.imwrite(annotated_img_path, annotated_img)  # 結果を画像として保存
        print(f"Saved annotated image for {image_file} to {annotated_img_path}")
