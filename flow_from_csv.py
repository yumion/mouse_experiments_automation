import numpy as np
import pandas as pd

def flow_from_csv(csv_file, batch_size, nb_classes, target_size, dimension=2, nb_frames=30, training=True):
    '''csvファイルからファイルパスとクラスインデックスを取得しgeneratorへ流す'''
    annotation_data = pd.read_csv(csv_file).values
    if training: # 訓練データはシャッフルする
        np.random.shuffle(annotation_data)
    while True:
        x, y = [], []
        i = 0
        for image_path, label_index in annotation_data:
            image = _set_input_data(image_path, target_size, dimension, nb_frames)
            y.append(label_index)
            x.append(image)
            i += 1
            if i == batch_size:
                yield (preprocess_input(np.array(x)), np.eye(nb_classes)[np.array(y)])
                i = 0
                x, y = [], []
