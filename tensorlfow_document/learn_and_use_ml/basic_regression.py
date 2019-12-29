# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/15 18:36

import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.utils import get_file

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'ModelYear', 'Origin']
column_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0]]
num_cols = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'ModelYear']
cate_cols = {'Origin': [1, 2, 3]}

dataset_path = get_file("auto-mpg.data",
                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
raw_df = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

train_df = raw_df.sample(frac=0.8, random_state=1234)
test_df = raw_df.drop(train_df.index)


def normalize_num_cols(df, num_columns):
    res_df = df.copy()
    for num_col in num_columns:
        max_value = df[num_col].max()
        min_value = df[num_col].min()
        res_df[num_col] = (df[num_col] - min_value) / (max_value - min_value + 1e-10)
    return res_df


train_df = normalize_num_cols(train_df, num_cols)
test_df = normalize_num_cols(test_df, num_cols)

train_df.to_csv("train_auto-mpg.csv", sep=',', header=False, index=False)
test_df.to_csv("test_auto-mpg.csv", sep=',', header=False, index=False)


def train_input_fn(data_file, batch_size=128):
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=column_defaults)
        features = dict(zip(column_names, columns))
        labels = features.pop('MPG')
        return features, labels

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.shuffle(100)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def eval_input_fn(data_file, batch_size=128):
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=column_defaults)
        features = dict(zip(column_names, columns))
        labels = features.pop('MPG')
        return features, labels

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def get_feature_columns():
    feature_cols = []

    for num_col in num_cols:
        feature_cols.append(tf.feature_column.numeric_column(num_col))
    for cate_col, cate_col_value in cate_cols.items():
        feature_cols.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(cate_col, cate_col_value)))
    return feature_cols


def dnn_regression_fn(features, labels, mode, params):
    top = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        # Add a hidden layer, densely connected on top of the previous layer.
        top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)
    out = tf.layers.dense(inputs=top, units=1, activation=None)
    prediction_value = tf.squeeze(out, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "pred_value": prediction_value,
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    loss = tf.losses.mean_squared_error(labels, prediction_value)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_spec = {
            'rmse': tf.metrics.root_mean_squared_error(labels, prediction_value),
            'mse': tf.metrics.mean_squared_error(labels, prediction_value),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    estimator = tf.estimator.Estimator(model_fn=dnn_regression_fn, model_dir=save_model_path,
                                       params={'feature_columns': get_feature_columns(), 'learning_rate': 0.001,
                                               "hidden_units": [64, 64]})
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn('train_auto-mpg.csv'), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn('test_auto-mpg.csv'), throttle_secs=120,
                                      start_delay_secs=120)
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("basic_dense_regression_model")


if __name__ == '__main__':
    tf.app.run()
