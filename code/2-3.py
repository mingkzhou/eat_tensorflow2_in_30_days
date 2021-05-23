# -*- coding: utf-8 -*-
"""
@author by mkzhou
@time 2021/5/23 下午2:43
"""

import tensorflow as tf
import numpy as np

def func1():
    x = tf.Variable(0.0, name='x', dtype=tf.float32)
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    with tf.GradientTape() as tape:
        y = a*tf.pow(x, 2) + b*x + c
    dy_dx = tape.gradient(y, x)
    print(dy_dx)


def func2():
    x = tf.Variable(0.0, name='x', dtype=tf.float32)
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for _ in range(1000):
        with tf.GradientTape() as tape:
            y = a*tf.pow(x, 2) + b*x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
    tf.print('y = {}; x = {}'.format(y, x))

def func3():
    x = tf.Variable(0.0, name='x', dtype=tf.float32)
    def f():
        a = tf.constant(1.0)
        b = tf.constant(-2.0)
        c = tf.constant(1.0)
        y = a*tf.pow(x, 2) + b*x + c
        return y

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for _ in range(1000):
        optimizer.minimize(f, [x])
    tf.print("y =", f(), "; x =", x)


if __name__ == '__main__':
    func3()