import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
# 自作関数
from flow_from_csv import flow_from_csv
from make_model import make_model

# annotaoinのパス
train_data_path = 'annotation_all_over3sec_train.csv'
test_data_path = 'annotation_all_over3sec_test.csv'

nb_classes = 2
batch_size = 8
nb_frames = 10 # 時系列解析で何フレーム見るか
# training
train_num_images = len(pd.read_csv(train_data_path).values)
train_steps_per_epoch = train_num_images // batch_size
# validation
test_num_images = len(pd.read_csv(test_data_path).values)
test_steps_per_epoch = test_num_images // batch_size

# data読み込み
train_gen = flow_from_csv(train_data_path,
                            batch_size=batch_size,
                            nb_classes=nb_classes,
                            target_size=[320, 240],
                            nb_frames=nb_frames,
                            shuffle=False)

test_gen = flow_from_csv(test_data_path,
                            batch_size=batch_size,
                            nb_classes=nb_classes,
                            target_size=[320, 240],
                            nb_frames=nb_frames,
                            shuffle=False)


# FC層のみ学習
model = make_model(nb_classes, nb_frames, 320, 240, 'VGG19', train_bottom=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
callbacks = []
callbacks.append(ModelCheckpoint(filepath='model_weights.h5', save_best_only=True, save_weights_only=True))
history = model.fit_generator(train_gen,
                steps_per_epoch=train_steps_per_epoch,
                epochs=30,
                callbacks=callbacks,
                validation_data=test_gen,
                validation_steps=test_steps_per_epoch,
                shuffle=False)

# 途中結果を表示
def plot_acc(history, save=False):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(range(1, len(acc)+1), acc, label='acc')
    plt.plot(range(1, len(val_acc)+1), val_acc, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    if save:
        plt.savefig('results_acc.png')
    plt.show()

plot_acc(history)

'''
# 全ての重みを更新
model = make_model(2, 30, 320, 240, 'VGG19', train_bottom=True)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['acc'])
callbacks = []
callbacks.append(ModelCheckpoint(filepath='model_weights_add_train.h5', save_best_only=True, save_weights_only=True))
history_add = model.fit_generator(train_gen,
                steps_per_epoch=len(train_gen),
                validation_data=test_gen,
                validation_steps=len(test_gen),
                epochs=100,
                callbacks=callbakcs,
                shuffle=False
                )

# 結果を表示
acc = history.history['acc'] + history_add.history['acc']
val_acc = history.history['val_acc'] + history_add.history['val_acc']
plt.plot(range(1, len(acc)+1), acc, label='acc')
plt.plot(range(1, len(val_acc)+1), val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('results_acc.png')
plt.show()
'''
