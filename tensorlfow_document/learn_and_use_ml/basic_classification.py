# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/15 17:51

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def create_model(input_shape=(784,), num_classes=10):
    img_input = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(img_input)
    x = layers.Dropout(0.5)(x)
    out_logits = layers.Dense(num_classes)(x)
    return Model(inputs=img_input, outputs=out_logits)


def model_fn(features, labels, mode, params):
    model = create_model()
    logits = model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(input=logits, axis=-1),
            "probabilities": tf.nn.softmax(logits, axis=-1),
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metric_spec = {
            'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(input=logits, axis=-1)),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=save_model_path, params={'learning_rate': 0.001, })
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(), throttle_secs=120, start_delay_secs=120)
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("basic_dense_model")


if __name__ == '__main__':
    tf.app.run()
