import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# クラス数
num_classes = 2

# ImageDataGenerator
train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            horizontal_flip=True,
            fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# base model
base_model = VGG19(weights='imagenet', include_top=False)
# add layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
classification = Dense(num_classes, activation='sigmoid', name='classification')(x)
# model
model = Model(inputs=base_model.input, outputs=classification)
model.summary()

# data読み込み
train_gen = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(320, 240),
        batch_size=128,
        class_mode='binary')

test_gen = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(320, 240),
        batch_size=128,
        class_mode='binary',
        shuffle=False)

# base modelの重みは更新しない(FC層のみ学習)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
callbacks = []
callbacks.append(ModelCheckpoint(filepath='model_weights.h5', save_best_only=True, save_weights_only=False))
history = model.fit_generator(train_gen,
                steps_per_epoch=len(train_gen),
                validation_data=test_gen,
                validation_steps=len(test_gen),
                epochs=10,
                shuffle=True
                )
# 途中結果を表示
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(range(1, len(acc)+1), acc, label='acc')
plt.plot(range(1, len(val_acc)+1), val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# 全ての重みを更新
for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['acc'])
callbacks = []
callbacks.append(ModelCheckpoint(filepath='model_weights_add_train.h5', save_best_only=True, save_weights_only=False))
history_add = model.fit_generator(train_gen,
                steps_per_epoch=len(train_gen),
                validation_data=test_gen,
                validation_steps=len(test_gen),
                epochs=50,
                shuffle=True
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
