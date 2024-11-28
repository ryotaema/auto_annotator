from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 学習済みモデルのパスを指定（例: 'best.pt'）
model_path = 'model/model_of_best/best.pt'      #モデルを変更してください
model = YOLO(model_path,task='detect')
# 推論する画像が入っているフォルダのパス
image_folder = 'data/pepper_data/image_1/color' #画像の入っているディレクトリに変更
# 出力アノテーションデータの保存先パス
output_folder = 'label/output/'                 #出力先のディレクトリに変更
os.makedirs(output_folder, exist_ok=True) 

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
            for result in results:  
                detections = result.boxes.data.cpu().numpy()  

                for det in detections:  
                    cls, x, y, w, h = int(det[5]), det[0], det[1], det[2], det[3]

                    # 0〜1の範囲に正規化
                    x /= img_width
                    y /= img_height
                    w /= img_width
                    h /= img_height

                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"Saved annotations for {image_file} to {output_path}")
