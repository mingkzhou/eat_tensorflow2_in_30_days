# -*- coding: utf-8 -*-
"""
@author by mkzhou
@time 2021/5/5 上午10:53
"""
"""
文本数据建模流程范例
"""
import re
import string
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

train_data_path = '../data/imdb/train.csv'
test_data_path = '../data/imdb/test.csv'
MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 32

def split_line(line):
    arr = tf.strings.split(line, '\t')
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int), axis=0)
    text = tf.expand_dims(arr[1], axis=1)
    return (text, label)

def get_data():
    ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path])\
        .map(split_line, num_parallel_calls=tf.data.experimental.AU)\
        .shuffle(buffer_size=1000).batch(BATCH_SIZE)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    ds_test_raw = tf.data.TextLineDataset(filenames = [test_data_path]) \
        .map(split_line,num_parallel_calls = tf.data.experimental.AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train_raw, ds_test_raw

def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,
                                                   '[%s]' % re.escape(string.punctuation), '')
    return cleaned_punctuation

class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):
        self.embedding = layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16, kernel_size=5, name='conv_1', activation='relu')
        self.pool_1 = layers.MaxPool1D(name='pool_1')
        self.conv_2 = layers.Conv1D(128, kernel_size=2, name='conv_2', activation='relu')
        self.pool_2 = layers.MaxPool1D(name='pool_2')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.pool_1(self.conv_1(x))
        x = self.pool_2(self.conv_2(x))
        x = self.dense(self.flatten(x))
        return x

    def summary(self, line_length=None, positions=None, print_fn=None):
        x_input = layers.Input(shape=MAX_LEN)
        output = self.call(x_input)
        model = tf.keras.Model(inputs=x_input, outputs=output)
        model.summary()

optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()
train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.BinaryAccuracy(name='train_accuracy')
valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)

@tf.function
def valid_step(model, features, labels):
    predictions = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)

def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs+1):
        for features, labels in ds_train:
            train_step(model, features, labels)
        for features, labels in ds_valid:
            valid_step(model, features, labels)
if __name__ == '__main__':
    model = CnnModel()
    model.build(input_shape=(None, MAX_LEN))
    model.summary()