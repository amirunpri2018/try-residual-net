# Implementation of ResNet-164
#
# Deep Residual Learning for Image Recognition
# https://arxiv.org/abs/1512.03385
#
# Identity Mappings in Deep Residual Networks
# https://arxiv.org/abs/1603.05027)# Wide Residual Network


import numpy as np
import pickle

from data_set                  import load_data
from funcy                     import concat, identity, juxt, partial, rcompose, repeat, repeatedly, take
from keras.callbacks           import LearningRateScheduler
from keras.layers              import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.models              import Model, save_model
from keras.optimizers          import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers        import l2
from keras.utils               import plot_model
from operator                  import getitem


def computational_graph(class_size):
    # Utility functions.

    def ljuxt(*fs):  # Kerasはジェネレーターを引数に取るのを嫌がるみたい、かつ、funcyはPython3だと積極的にジェネレーターを使うみたいなので、リストを返すjuxtを作りました。
        return rcompose(juxt(*fs), list)

    def batch_normalization():
        return BatchNormalization()

    def relu():
        return Activation('relu')

    def conv(filter_size, kernel_size, stride_size=1):
        return Conv2D(filter_size, kernel_size, strides=stride_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001), use_bias=False)

    def add():
        return Add()

    def global_average_pooling():
        return GlobalAveragePooling2D()

    def dense(unit_size, activation):
        return Dense(unit_size, activation=activation, kernel_regularizer=l2(0.0001))

    # Define ResNet-164

    def first_residual_unit(filter_size, stride_size):
        return rcompose(batch_normalization(),
                        relu(),
                        ljuxt(rcompose(conv(filter_size // 4, 1, stride_size),
                                       batch_normalization(),
                                       relu(),
                                       conv(filter_size // 4, 3),
                                       batch_normalization(),
                                       relu(),
                                       conv(filter_size, 1)),
                              rcompose(conv(filter_size, 1, stride_size))),
                        add())

    def residual_unit(filter_size):
        return rcompose(ljuxt(rcompose(batch_normalization(),
                                       relu(),
                                       conv(filter_size // 4, 1),
                                       batch_normalization(),
                                       relu(),
                                       conv(filter_size // 4, 3),
                                       batch_normalization(),
                                       relu(),
                                       conv(filter_size, 1)),
                              identity),
                        add())

    def residual_block(filter_size, stride_size, unit_size):
        return rcompose(first_residual_unit(filter_size, stride_size),
                        rcompose(*repeatedly(partial(residual_unit, filter_size), unit_size - 1)))

    n = 18  # ResNet-164の164は、1（最初） + 3 * n * 3 + 1（最後）みたい。

    return rcompose(conv(16, 3),
                    residual_block( 64, 1, n),
                    residual_block(128, 2, n),
                    residual_block(256, 2, n),
                    batch_normalization(),
                    relu(),
                    global_average_pooling(),
                    dense(class_size, 'softmax'))


def main():
    (x_train, y_train), (x_validation, y_validation) = load_data()

    model = Model(*juxt(identity, computational_graph(y_train.shape[1]))(Input(shape=x_train.shape[1:])))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='./results/model.png')

    train_data      = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
    validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    for data in (train_data, validation_data):
        data.fit(x_train)  # 実用を考えると、x_validationでのfeaturewiseのfitは無理だと思う……。

    batch_size = 128
    epoch_size = 200

    results = model.fit_generator(train_data.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epoch_size,
                                  callbacks=[LearningRateScheduler(partial(getitem, tuple(take(epoch_size, concat(repeat(0.1, 80), repeat(0.01, 42), repeat(0.001))))))],
                                  validation_data=validation_data.flow(x_validation, y_validation, batch_size=batch_size),
                                  validation_steps=x_validation.shape[0] // batch_size)

    with open('./results/history.pickle', 'wb') as f:
        pickle.dump(results.history, f)

    save_model(model, './results/model.h5')

    del model


if __name__ == '__main__':
    main()
