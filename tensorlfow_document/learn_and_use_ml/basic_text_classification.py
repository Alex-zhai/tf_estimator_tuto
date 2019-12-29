# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/15 18:14

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

max_features = 10000
seq_len = 256
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=seq_len)
x_test = pad_sequences(x_test, maxlen=seq_len)


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


def create_model(input_shape=(seq_len,), num_classes=2):
    img_input = layers.Input(shape=input_shape)
    x = layers.Embedding(max_features, 16)(img_input)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
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
    train_and_eval("basic_text_dense_model")


if __name__ == '__main__':
    model = create_model()
    print(model.summary())
    tf.app.run()
