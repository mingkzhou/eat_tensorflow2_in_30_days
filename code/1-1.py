# -*- coding: utf-8 -*-
"""
@author by mkzhou
@time 2021/4/30 下午5:38
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras import models, layers
matplotlib.use('TkAgg')

# 数据处理
def preprocessing(dfdata):
    dfresult = pd.DataFrame()
    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return(dfresult)

# read data
def read_data():
    dftrain_raw = pd.read_csv('../data/titanic/train.csv')
    dftest_raw = pd.read_csv('../data/titanic/test.csv')
    ax = dftrain_raw['Survived'].value_counts().plot(
        kind='bar', figsize=(12, 8), fontsize=15, rot=0)
    ax.set_ylabel('Counts', fontsize=15)
    ax.set_xlabel('Survived', fontsize=15)
    # plt.show()
    plt.close()
    x_train = preprocessing(dftrain_raw)
    y_train = dftrain_raw['Survived'].values

    x_test = preprocessing(dftest_raw)
    y_test = dftest_raw['Survived'].values

    print("x_train.shape =", x_train.shape)
    print("x_test.shape =", x_test.shape)
    return x_train, y_train

def getmodel():
    tf.keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu', input_shape=(15,)))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

def train_model():
    model = getmodel()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    x_train, y_train = read_data()
    history = model.fit(x_train, y_train,
                        batch_size=64, epochs=30,
                        validation_split=0.2)
    return history

def plot_metric(history, metric):
    train_metric = history.history[metric]
    val_metric = history.history['val_'+metric]
    epochs = range(1, len(train_metric)+1)
    plt.plot(epochs, train_metric, 'bo--')
    plt.plot(epochs, val_metric, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

def save_model(model, model_path):
    # 保存模型及权重
    model.save(model_path)
    # 加载模型
    models.load_model(model_path)
    # 保存模型结构
    json_str = model.to_json()
    # 恢复模型结构
    model_json = models.model_from_json(json_str)
    # 保存模型权重
    model.save_weights(model_path)
    # 恢复模型结构
    model_json.load_weights(model_path)

if __name__ == '__main__':
    # model = getmodel()
    history = train_model()
    plot_metric(history, 'loss')


