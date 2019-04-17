import numpy as np
import re
import os
import pandas as pd
from collections import Counter
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical


def flow_from_csv(csv_file, nb_classes, batch_size, target_size, nb_frames=10, shuffle=True):
    '''csvファイルからファイルパスとクラスインデックスを取得しgeneratorへ流す'''
    annotation_data = pd.read_csv(csv_file).values
    if shuffle: # 訓練データはシャッフルする
        np.random.shuffle(annotation_data)
    while True:
        x, y = [], []
        i = 0
        frame_x, frame_y = [], []
        frame_count = 0
        for image_path, label_index in annotation_data:
            # 1batchで流すフレーム数を貯める
            image = load_img(image_path, target_size=target_size)
            image = img_to_array(image)
            frame_x.append(image)
            frame_y.append(label_index)
            frame_count += 1
            if frame_count == nb_frames:
                # nb_franes枚溜まったら1batchでまとめる
                x.append(np.array(frame_x))
                # nb_frames枚の中で再頻出のラベルを代表として教師ラベルとして渡す
                counter = Counter(frame_y) # ユニークキーでカウント
                frame_y_max = counter.most_common()[0][0] # 最頻出の値を取得
                y.append(np.array(frame_y_max))
                i += 1
                frame_x, frame_y = [], []
                frame_count = 0
                if i == batch_size:
                    flow_x = _preprocess_input(np.array(x))
                    flow_y = to_categorical(np.array(y), nb_classes)
                    yield (flow_x, flow_y)
                    i = 0
                    x, y = [], []


def _preprocess_input(x_array):
    '''入力画像を規格化'''
    # 入力画像の処理
    if len(x_array.shape) == 3:  # (height, width, channel)
        x_array = np.expand_dims(x_array, axis=0)  # (1, height, width, channel)
    x_array = x_array.astype('float32')
    x_array /= 255
    return x_array
