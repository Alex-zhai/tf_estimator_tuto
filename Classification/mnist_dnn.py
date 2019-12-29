# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/3/8 10:31


from __future__ import absolute_import, print_function, division

import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist

img_col = 28
img_row = 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
x_train /= 255.0
x_test /= 255.0


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({"feats": x_train}, y_train))
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({"feats": x_test}, y_test))
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def get_feature_columns():
    return [tf.feature_column.numeric_column(key='feats')]


def train_and_eval(save_model_dir):
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)
    config = tf.estimator.RunConfig(model_dir="mnist_cnn", save_checkpoints_steps=1000,
                                    train_distribute=distribution)
    estimator = tf.estimator.DNNClassifier(hidden_units=[256, 128, 64], feature_columns=get_feature_columns(),
                                           model_dir=save_model_dir, config=config)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    start_time = time.time()
    save_model_dir = "dnn_mnist"
    train_and_eval(save_model_dir)
    print("runing time is ", time.time() - start_time)


if __name__ == '__main__':
    tf.app.run()
