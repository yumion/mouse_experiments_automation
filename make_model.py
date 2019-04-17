from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, LSTM, Input, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.applications import *
import tensorflow as tf

def make_model(nb_classes, frame_per_batch, width, height, back_bone='VGG19', train_bottom=False):
    with tf.device('/cpu:0'):
        # base model
        if back_bone == 'VGG19':
            base_model = vgg19.VGG19(weights='imagenet', include_top=False)
        elif back_bone == 'Xception':
            base_model = xception.Xception(weights='imagenet', include_top=False)
        elif back_bone == 'ResNet50':
            base_model = resnet50.ResNet50(weights='imagenet', include_top=False)
        elif back_bone == 'IncevtionV3':
            base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        # LSTM architecture
        video_input = Input(shape=(frame_per_batch, width, height, 3))
        x = TimeDistributed(base_model)(video_input)
        x = TimeDistributed(Flatten())(x)
        x = Dense(512, activation='relu')(x)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        classification = Dense(nb_classes, activation='sigmoid', name='classification')(x)
        # make model
        model = Model(inputs=video_input, outputs=classification)
        # base modelの重み更新
        for layer in base_model.layers:
            layer.trainable = train_bottom
        model.summary()

    return model
model = make_model(2, 10, 320, 240, 'VGG19', train_bottom=False)
