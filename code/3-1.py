# -*- coding: utf-8 -*-
"""
@author by mkzhou
@time 2021/5/23 下午3:00
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def get_data():
    n = 400
    x = tf.random.uniform([n, 2], minval=-10, maxval=10)
    w0 = tf.constant([[2.0], [-3.0]])
    b0 = tf.constant([[3.0]])

    # @表示矩阵乘法
    y = x@w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)

    return x, y

def data_iter(features, labels, batch_size=8):
    num = len(features)
    index = list(range(num))
    np.random.shuffle(index)
    for i in range(0, num, batch_size):
        indexs = index[i: min(i + batch_size, num)]
        yield tf.gather(features, indexs), tf.gather(labels, indexs)

class linear_model():
    # 正向传播
    def __call__(self, x):
        w = tf.Variable(tf.random.normal((2, 2)))
        b = tf.Variable(tf.zeros_like([[3.0]], dtype=tf.float32))
        return x@w + b
    def loss_func(self, y_true, y_pred):
        return tf.reduce_mean((y_true - y_pred)**2/2)

def train():
    model = linear_model()
    x, y = get_data()
    for i in range(200):
        for features, labels in data_iter(x, y, 10):
            with tf.GradientTape() as tape:
                pre = model(features)
                loss = model.loss_func(y_true=labels, y_pred=pre)

def gen_data():
    #正负样本数量
    n_positive, n_negative = 2000, 2000

    #生成正样本, 小圆环分布
    r_p = 5.0 + tf.random.truncated_normal([n_positive, 1], 0.0, 1.0)
    theta_p = tf.random.uniform([n_positive, 1], 0.0, 2*np.pi)
    Xp = tf.concat([r_p*tf.cos(theta_p), r_p*tf.sin(theta_p)], axis=1)
    Yp = tf.ones_like(r_p)

    #生成负样本, 大圆环分布
    r_n = 8.0 + tf.random.truncated_normal([n_negative, 1], 0.0, 1.0)
    theta_n = tf.random.uniform([n_negative, 1], 0.0, 2*np.pi)
    Xn = tf.concat([r_n*tf.cos(theta_n), r_n*tf.sin(theta_n)], axis=1)
    Yn = tf.zeros_like(r_n)

    #汇总样本
    X = tf.concat([Xp, Xn], axis=0)
    Y = tf.concat([Yp, Yn], axis=0)
    return X, Y

# 构建数据管道迭代器
def iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = indices[i: min(i + batch_size, num_examples)]
        yield tf.gather(features, indexs), tf.gather(labels, indexs)

class DNNModel(tf.Module):
    def __init__(self, name=None):
        super(DNNModel, self).__init__(name=name)
        self.w1 = tf.Variable(tf.random.truncated_normal([2, 4]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([1, 4]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.truncated_normal([4, 8]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1, 8]), dtype=tf.float32)
        self.w3 = tf.Variable(tf.random.truncated_normal([8, 1]), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.nn.relu(x@self.w1 + self.b1)
        x = tf.nn.relu(x@self.w2 + self.b2)
        y = tf.nn.sigmoid(x@self.w3 + self.b3)
        return y

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def loss_func(self, y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)
        # bce = tf.keras.losses.BinaryCrossentropy(y_true, y_pred)
        bce = -y_true * tf.math.log(y_pred) - (1-y_true)*tf.math.log(1-y_pred)
        return tf.reduce_mean(bce)

    # 评估指标(准确率)
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def metric_func(self, y_true, y_pred):
        y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred, dtype=tf.float32),
                          tf.zeros_like(y_pred, dtype=tf.float32))
        acc = tf.reduce_mean(1 - tf.abs(y_true - y_pred))
        return acc

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        pre = model(features)
        loss = model.loss_func(labels, pre)

    grad = tape.gradient(loss, model.trainable_variables)
    for p, dloss_dp in zip(model.trainable_variables, grad):
        p.assign(p - 0.01*dloss_dp)
    # 计算评估指标
    metric = model.metric_func(labels, pre)

    return loss, metric

def train_model():
    model = DNNModel()
    X, Y = gen_data()
    for epoch in tf.range(1, 1001):
        for features, labels in iter(X, Y, 100):
            loss, metric = train_step(model, features, labels)
        if epoch%100 == 0:
            tf.print("epoch =", epoch, "loss = ", loss, "accuracy = ", metric)

if __name__ == '__main__':
    train_model()