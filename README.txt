note: dataset中的影像皆經過YOLOR裁切

1. 執行gen_data.py產生訓練及測試*.npy於npys目錄
2. (選) train.py訓練及評估多標籤分類模型
3. train_embed.py訓練及評估Triplet模型，產生model_embed.h5
4. 下載yolor_p6.pt (https://drive.google.com/file/d/1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76/view)
5. predict.py 讀取__test__目錄中的影像，產生predict.jpg，綠色框代表校狗，紅色框代表非校狗

