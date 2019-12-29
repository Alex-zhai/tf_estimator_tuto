# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/1/28 17:06

from __future__ import absolute_import, print_function, division
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import mnist

img_col = 28
img_row = 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_col, img_row, 1)
x_test = x_test.reshape(x_test.shape[0], img_col, img_row, 1)
# x_train = x_train[:100]
# y_train = y_train[:100]
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
x_train /= 255.0
x_test /= 255.0


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def create_model():
    inp = layers.Input(shape=[img_row, img_col, 1])
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(10)(x)
    return Model(inputs=inp, outputs=out)


def create_dense_model():
    inp = layers.Input(shape=[img_row, img_col, 1])
    x = layers.Flatten()(inp)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(10)(x)
    return Model(inputs=inp, outputs=out)


def model_fn(features, labels, mode):
    cnn_model = create_model()
    logits = cnn_model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(logits, -1),
            "probabilities": tf.nn.softmax(logits, -1)
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(logits, -1))
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, eval_metric_ops=eval_metric_ops, loss=loss)


def train_and_eval():
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)
    config = tf.estimator.RunConfig(model_dir="mnist_cnn", save_checkpoints_steps=1000,
                                    train_distribute=distribution)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    start_time = time.time()
    train_and_eval()
    print("runing time is ", time.time() - start_time)


if __name__ == '__main__':
    model = create_dense_model()
    print(model.summary())
    # tf.app.run()
