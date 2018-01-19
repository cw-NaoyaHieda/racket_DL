#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:59:55 2018

@author: naoya
"""
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras import optimizers
# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=X_train.shape[1:])
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))

# 最後のconv層の直前までの層をfreeze
for layer in model.layers[:15]:
    layer.trainable = False

# Fine-tuningのときはSGDの方がよい
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
plot_model(model,to_file="vgg16.png")

history = model.fit(X_train, Y_train,
                    batch_size=128,
                    nb_epoch=100,
                    verbose=1,
                    validation_split=0.1)

# 学習履歴をプロット
plot_history(history)



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
