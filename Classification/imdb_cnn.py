# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/1/28 16:40

from __future__ import print_function, division, absolute_import
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

seq_len = 80
max_features = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, seq_len)
x_test = pad_sequences(x_test, seq_len)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def create_model():
    inp = layers.Input(shape=(seq_len,))
    x = layers.Embedding(input_dim=max_features, output_dim=128)(inp)
    x = layers.Dropout(0, 2)(x)
    x = layers.Conv1D(250, kernel_size=3, strides=1, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(250, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(2)(x)
    return Model(inputs=inp, outputs=out)


def cnn_model_fn(features, labels, mode):
    cnn_model = create_model()
    logits = cnn_model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=-1),
            'probabilities': tf.nn.softmax(logits, axis=-1)
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    elif mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metric_spec = {
            'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(input=logits, axis=-1)),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    cnn_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=save_model_path
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=cnn_estimator, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("imdb_cnn_model")

if __name__ == '__main__':
    model = create_model()
    print(model.summary())
    tf.app.run()