from glob import glob
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


# ImageDataGenerator
train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=90,
            horizontal_flip=True,
            fill_mode='constant')

test_datagen = ImageDataGenerator(rescale=1./255)

# base model
# Xception
# from keras.applications.xception import Xception
# base_model = Xception(weights='imagenet', include_top=False)
# VGG
from keras.applications.vgg19 import VGG19
base_model = VGG19(weights='imagenet', include_top=False)
# add layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
classification = Dense(1, activation='sigmoid', name='classification')(x)
# model
model = Model(inputs=base_model.input, outputs=classification)
model.summary()

# data読み込み
train_gen = train_datagen.flow_from_directory(
        'downsampling/train',
        target_size=(320, 240),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

test_gen = test_datagen.flow_from_directory(
        'downsampling/test',
        target_size=(320, 240),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

# base modelの重みは更新しない(FC層のみ学習)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
callbacks = []
callbacks.append(ModelCheckpoint(filepath='model_weights.h5', save_best_only=True, save_weights_only=True))
history = model.fit_generator(train_gen,
                steps_per_epoch=len(train_gen),
                validation_data=test_gen,
                validation_steps=len(test_gen),
                epochs=100,
                callbacks=callbacks,
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
callbacks.append(ModelCheckpoint(filepath='model_weights_add_train.h5', save_best_only=True, save_weights_only=True))
history_add = model.fit_generator(train_gen,
                steps_per_epoch=len(train_gen),
                validation_data=test_gen,
                validation_steps=len(test_gen),
                epochs=100,
                callbacks=callbacks,
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
