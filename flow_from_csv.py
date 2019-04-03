import numpy as np
import re
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array


def flow_from_csv(csv_file, batch_size, nb_classes, target_size, dimension=2, nb_frames=30, shuffle=True):
    '''csvファイルからファイルパスとクラスインデックスを取得しgeneratorへ流す'''
    annotation_data = pd.read_csv(csv_file).values
    if shuffle: # 訓練データはシャッフルする
        np.random.shuffle(annotation_data)
    while True:
        x, y = [], []
        i = 0
        for image_path, label_index in annotation_data:
            image = _set_input_data(image_path, target_size, dimension, nb_frames)
            x.append(image)
            y.append(label_index)
            i += 1
            if i == batch_size:
                yield (_preprocess_input(np.array(x), np.array(y)))
                i = 0
                x, y = [], []


def _set_input_data(image_path, target_size, dimension, nb_frames):
    # color order: (R, G, B)
    if dimension == 2: # frame
        image = load_img(image_path, target_size=target_size)
        array = img_to_array(image)

    elif dimension == 3: # video
        array = []
        first_frame_index = int(re.findall('(\d+)', image_path)[-1])
        # 指定したフレーム分スタックする
        for frame_index in range(first_frame_index, first_frame_index + nb_frames):
            sequence_path = image_path.replace(str(first_frame_index), str(frame_index))
            frame = load_img(sequence_path, target_size=target_size)
            array.append(img_to_array(frame))
        array = np.asarray(array)

    else:
        raise ValueError('invalid dimension arg. dimension should be 2 or 3')

    return array


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
