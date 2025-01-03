# auto_annotator
This is an automatic annotation code created to recognize green peppers.

## 事前準備
```
pip install ultralytics
pip install --upgrade ultralytics
```
```
git clone https://github.com/ryotaema/auto_annotator.git

cd auto_annotator
mkdir -p model/best_model
mkdir -p label/output/label_image
```
## 使用方法
1. best_modelの中に学習済みモデルを入れてください。

2. model_path、image_folder、output_folderのパスを指定する。
 *  model_path   : 学習済みモデルのパス  (ex:best.pt)
 *  image_folder : 推論画像のフォルダパス 
 *  output_folder: アノテーションデータの出力先フォルダ 
 *  label_image: ラベルのついた画像の出力先\
※　output_folderはなかったら勝手に生成します。


3. （＃推論実行）のresults = model.predict(image_path, conf = 0.75)で推論の設定 
上の例ではconf = 0.75で最小信頼度の閾値を0.75に設定している。 
以下のサイトに推論の引数がまとめられています。 \
「Ultralytics YOLO Docs」推論 \
https://docs.ultralytics.com/ja/modes/predict/#inference-sources:~:text=of%20Results%20objects-,%E6%8E%A8%E8%AB%96,-model.predict() 

4. 実行
```
python3 auto_labelimg.py
```
