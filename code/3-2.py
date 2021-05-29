# -*- coding: utf-8 -*-
"""
@author by mkzhou
@time 2021/5/23 下午4:57
"""
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, optimizers

def get_data():
    #样本数量
    n = 400
    # 生成测试用数据集
    X = tf.random.uniform([n, 2], minval=-10, maxval=10)
    w0 = tf.constant([[2.0], [-3.0]])
    b0 = tf.constant([[3.0]])
    Y = X@w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)  # @表示矩阵乘法,增加正态扰动
    return X, Y

def get_model():
    model = layers.Dense(units=1)
    model.build(input_shape=(2, ))
    model.loss_func = losses.mean_squared_error
    model.optimizer = optimizers.SGD(learning_rate=0.001)
    return model

#使用autograph机制转换成静态图加速
@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1]))
    grads = tape.gradient(loss, model.variables)
    model.optimizer.apply_gradients(zip(grads, model.variables))
    return loss

@tf.function
def train_model():
    X, Y = get_data()
    ds = tf.data.Dataset.from_tensor_slices((X, Y))\
        .shuffle(buffer_size=100)\
        .batch(10)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    model = get_model()
    for epoch in tf.range(1, 201):
        loss = tf.constant(0.0)
        for features, labels in ds:
            loss = train_step(model, features, labels)
        if epoch%50 == 0:
            tf.print("epoch =", epoch, "loss = ", loss)
            # tf.print("w =", model.variables[0])
            # tf.print("b =", model.variables[1])

if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    train_model()

