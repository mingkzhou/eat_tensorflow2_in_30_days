# -*- coding: utf-8 -*-
"""
@author by mkzhou
@time 2021/5/9 下午12:06
"""
"""
时间序列数据建模
"""
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, callbacks

matplotlib.use('TkAgg')
WINDOW_SIZE = 8
def get_data():
    df = pd.read_csv('../data/covid-19.csv', sep='\t')
    # df.plot(x = "date",y = ["confirmed_num","cured_num","dead_num"],figsize=(10,6))
    # plt.xticks(rotation=60)
    # plt.show()
    dfdata = df.set_index('date')
    dfdiff = dfdata.diff(periods=1).dropna()
    dfdiff = dfdiff.reset_index('date')
    dfdiff = dfdiff.drop('date', axis=1).astype('float32')

    ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values, dtype=tf.float32)) \
        .window(WINDOW_SIZE, shift=1).flat_map(batch_dataset)

    ds_label = tf.data.Dataset.from_tensor_slices(
        tf.constant(dfdiff.values[WINDOW_SIZE:], dtype=tf.float32))

    #数据较小，可以将全部训练数据放入到一个batch中，提升性能
    ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(38).cache()
    return ds_train

def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    return dataset_batched

class Block(layers.Layer):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)

    def call(self, x_input, x):
        x_out = tf.maximum((1+x)**x_input[:, -1, :], 0.0)
        return x_out

    def get_config(self):
        config = super(Block, self).get_config()
        return config

def get_model():
    tf.keras.backend.clear_session()
    x_input = layers.Input(shape=(None, 3), dtype=tf.float32)
    x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x_input)
    x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
    x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
    x = layers.LSTM(3, input_shape=(None, 3))(x)
    x = layers.Dense(3)(x)
    x = Block()(x_input, x)
    model = models.Model(inputs=[x_input], outputs=[x])
    model.summary()
    return model

class MSPE(losses.Loss):
    def call(self, y_true, y_pred):
        err_percent = (y_true - y_pred)**2 / (tf.maximum(y_true**2, 1e-7))
        mean_err_percent = tf.reduce_mean(err_percent)
        return mean_err_percent
    def get_config(self):
        config = super(MSPE, self).get_config()
        return config

def train():
    ds_train = get_data()

    model = get_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=MSPE(name='MSPE'))

    # 如果loss在100个epoch后没有提升，则学习了减半
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=100)
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)
    callbacks_list = [lr_callback, stop_callback]

    history = model.fit(ds_train, epochs=500, callbacks=callbacks_list)


if __name__ == '__main__':
    # get_model()
    # get_data()
    train()
