import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array


def flow_from_csv(csv_file, batch_size, nb_classes, target_size, shuffle=True):
    '''csvファイルからファイルパスとクラスインデックスを取得しgeneratorへ流す'''
    annotation_data = pd.read_csv(csv_file).values
    if shuffle: # 訓練データはシャッフルする
        np.random.shuffle(annotation_data)
    while True:
        x, y = [], []
        i = 0
        for image_path, label_index in annotation_data:
            image = load_img(image_path, target_size)
            image = img_to_array(image)
            y.append(label_index)
            x.append(image)
            i += 1
            if i == batch_size:
                yield (_preprocess_input(np.array(x)), np.eye(nb_classes)[np.array(y)])
                i = 0
                x, y = [], []


def _preprocess_input(x_array, y_array):
    '''入力画像を規格化、ラベルをone-hotに'''
    # 入力画像の処理
    if len(x_array.shape) == 3:  # (height, width, channel)
        x_array = np.expand_dims(x_array, axis=0)  # (1, height, width, channel)
    x_array = x_array.astype('float32')
    x_array /= 255
    # ラベルの処理
    y_array = y_array.ravel()  # (num_samples, 1) => (num_samples, )
    if len(y_array.shape) == 1:  # (num_samples, )
        num_classes = np.max(y_array) + 1
        y_array = np.eye(num_classes)[y_array]  # one hot: (num_samples, num_classes)
    y_array = y_array.astype('float32')
    return x_array, y_array
