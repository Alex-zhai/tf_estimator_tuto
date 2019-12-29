# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/1/30 10:24

from __future__ import absolute_import, division, print_function

import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import layers

# tf.enable_eager_execution()

CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

NUM_COLS = ['age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num']

CATE_COLS = ['gender', 'race', 'education', 'marital_status', 'relationship',
             'workclass', 'occupation', 'native_country']
LABEL_COL = 'income_bracket'
LABEL_CONS = ['>50K', '<=50K']
WEIGHT_COL = 'fnlwgt'

train_df = pd.read_csv("adult.data.csv", header=None, names=CSV_COLUMNS)
test_df = pd.read_csv("adult.test.csv", header=None, names=CSV_COLUMNS)


# print(train_df.info())
# print(train_df.head(5))


def get_cate_feat_values():
    cate_feat_values = {}
    for cate_col in CATE_COLS:
        cate_feat_values[cate_col] = train_df[cate_col].unique()
    return cate_feat_values


def train_input_fn(train_csv_path, batch_size=32, epochs=10):
    dataset = tf.contrib.data.make_csv_dataset(file_pattern=train_csv_path, batch_size=batch_size,
                                               column_names=CSV_COLUMNS,
                                               column_defaults=_CSV_COLUMN_DEFAULTS, label_name=LABEL_COL,
                                               field_delim=',', use_quote_delim=True, header=False,
                                               num_epochs=epochs, shuffle=True)
    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()

    return features, target


def eval_input_fn(train_csv_path, batch_size=32, epochs=1):
    dataset = tf.contrib.data.make_csv_dataset(file_pattern=train_csv_path, batch_size=batch_size,
                                               column_names=CSV_COLUMNS,
                                               column_defaults=_CSV_COLUMN_DEFAULTS, label_name=LABEL_COL,
                                               field_delim=',', use_quote_delim=True, header=False,
                                               num_epochs=epochs, shuffle=False)
    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, target


def get_feature_columns():
    feat_columns = []
    for num_col in NUM_COLS:
        feat_columns.append(tf.feature_column.numeric_column(num_col))
    for cate_col_name, cata_col_values in get_cate_feat_values().items():
        feat_columns.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(cate_col_name, cata_col_values)))
    return feat_columns


def model_inference(input_x):
    feature_columns = get_feature_columns()
    input_layer = tf.feature_column.input_layer(input_x, feature_columns)
    inp = input_layer
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out_logit = layers.Dense(1, activation=None)(x)
    return out_logit


def model_fn(features, labels, mode):
    logits = model_inference(features)
    head = tf.contrib.estimator.binary_classification_head(label_vocabulary=LABEL_CONS, weight_column=WEIGHT_COL)
    return head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels,
        optimizer=tf.train.AdamOptimizer(0.001)
    )


def train_and_eval():
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='income_pred_model')
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn("adult.data.csv"), max_steps=100000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn("adult.test.csv"))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval()


if __name__ == '__main__':
    # features, target = train_input_fn("adult.data.csv")
    # input_layer = tf.feature_column.input_layer(features, get_feature_columns())
    # print(input_layer)
    # inference = model_inference(features)
    # print(inference)
    tf.app.run()

