#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:11:11 2018

@author: naoya
"""

# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from keras import optimizers
import time

def plot_cifar10(X, y, result_dir):
    plt.figure()

    # 画像を描画
    nclasses = 10
    pos = 1
    for targetClass in range(nclasses):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(y)):
            if y[i][0] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        np.random.shuffle(targetIdx)
        for idx in targetIdx[:10]:
            img = toimage(X[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.savefig(os.path.join(result_dir, 'plot.png'))


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))
            
def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


# CIFAR-10データセットをロード
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# 入力画像の次元
img_rows, img_cols = 32, 32

# チャネル数（RGBなので3）
img_channels = 3

# CIFAR-10データをロード
# (nb_samples, nb_rows, nb_cols, nb_channel) = tf
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# ランダムに画像をプロット
# plot_cifar10(X_train, y_train, result_dir)

# 画素値を0-1に変換
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

nclasses = 10
# クラスラベル（0-9）をone-hotエンコーディング形式に変換
Y_train = np_utils.to_categorical(y_train, nclasses)
Y_test = np_utils.to_categorical(y_test, nclasses)

# CNNを構築
model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nclasses))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

resultdir = "/Desktop/notebook/assignment/result"
# モデルのサマリを表示
model.summary()
plot_model(model,to_file="tmp.png")

start = time.time()
history = model.fit(X_train, Y_train,
                    batch_size=128,
                    nb_epoch=100,
                    verbose=1,
                    validation_split=0.1)
elapsed_time = time.time() - start
# 学習履歴をプロット
plot_history(history)


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

input_tensor = Input(shape=X_train.shape[1:])
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

# VGG16とFCを接続
model_vgg16 = Model(input=vgg16.input, output=top_model(vgg16.output))

# 最後のconv層の直前までの層をfreeze
for layer in model_vgg16.layers[:15]:
    layer.trainable = False

# Fine-tuningのときはSGDの方がよい
model_vgg16.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model_vgg16.summary()
plot_model(model_vgg16,to_file="vgg16.png")
start2 = time.time()
history_vgg16 = model_vgg16.fit(X_train, Y_train,
                    batch_size=128,
                    nb_epoch=100,
                    verbose=1,
                    validation_split=0.1)
elapsed_time2 = time.time() - start2
# 学習履歴をプロット
plot_history(history_vgg16)




score = model_vgg16.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print ("elapsed_time:{0}".format(elapsed_time2) + "[sec]")

