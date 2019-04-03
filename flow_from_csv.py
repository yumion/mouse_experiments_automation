import numpy as np
import re
import os
import pandas as pd
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
            image = load_img(image_path, target_size=target_size)
            image = img_to_array(image)
            frame_x.append(image)
            # frame_y.append(label_index)
            frame_count += 1
            if frame_count == nb_frames:
                x.append(np.array(frame_x))
                # y.append(np.array(frame_y))
                y.append(np.array(label_index))
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
