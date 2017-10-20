import numpy as np
import pickle

from data_set                  import load_data
from funcy                     import concat, identity, juxt, partial, rcompose, repeat, repeatedly, take
from keras.callbacks           import LearningRateScheduler
from keras.layers              import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.models              import Model, save_model
from keras.optimizers          import SGD
from keras.preprocessing.image import ImageDataGenerator
from operator                  import getitem


def ljuxt(*fs):  # Kerasはジェネレーターを引数に取るのを嫌がるみたい、かつ、funcyはPython3だと積極的にジェネレーターを使うみたいなので、リストを返すjuxtを作りました。
    return rcompose(juxt(*fs), list)


def create_model(x, y):
    def conv_2d(filters, kernel_size, strides=1):
        return Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)

    def bn_relu():
        return rcompose(BatchNormalization(), Activation('relu'))

    def bn_relu_conv(filters, kernel_size, strides=1):
        return rcompose(bn_relu(), conv_2d(filters, kernel_size, strides))

    def first_residual_unit(filters, strides):
        return rcompose(bn_relu(),
                        ljuxt(rcompose(conv_2d(filters // 4, 1, strides),
                                       bn_relu(),
                                       conv_2d(filters // 4, 3),
                                       bn_relu(),
                                       conv_2d(filters,      1)),
                              rcompose(conv_2d(filters,      1, strides))),
                        Add())

    def residual_unit(filters):
        return rcompose(ljuxt(rcompose(bn_relu_conv(filters // 4, 1),
                                       bn_relu_conv(filters // 4, 3),
                                       bn_relu_conv(filters,      1)),
                              identity),
                        Add())

    def residual_units(filters, strides):
        return rcompose(first_residual_unit(filters, strides),
                        rcompose(*repeatedly(partial(residual_unit, filters), 18 - 1)))

    computational_graph = rcompose(conv_2d(16, 3),
                                   residual_units( 64, 1),
                                   residual_units(128, 2),
                                   residual_units(256, 2),
                                   BatchNormalization(),
                                   Activation('relu'),
                                   GlobalAveragePooling2D(),
                                   Dense(y.shape[1], activation='softmax'))

    return Model(*juxt(identity, computational_graph)(Input(shape=x.shape[1:])))


def main():
    (x_train, y_train), (x_validation, y_validation) = load_data()

    model = create_model(x_train, y_train)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.9, decay=0.0001, nesterov=True),
                  metrics=['accuracy'])

    model.summary()

    train_data      = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
    validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    for data in (train_data, validation_data):
        data.fit(x_train)  # 実用を考えると、x_validationでのfeaturewiseのfitは無理だと思う……。

    batch_size = 64
    epochs     = 200

    results = model.fit_generator(train_data.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  callbacks=[LearningRateScheduler(partial(getitem, tuple(take(epochs, concat(repeat(0.01, 1), repeat(0.1, 79), repeat(0.01, 42), repeat(0.001))))))],
                                  validation_data=validation_data.flow(x_validation, y_validation, batch_size=batch_size),
                                  validation_steps=x_validation.shape[0] // batch_size,
                                  workers=4)

    with open('./history.pickle', 'wb') as f:
        pickle.dump(results.history, f)

    save_model(model, './model.h5')

    del model


if __name__ == '__main__':
    main()
