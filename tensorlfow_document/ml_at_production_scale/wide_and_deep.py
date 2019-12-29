# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/18 11:21

# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/18 9:24

from __future__ import print_function, division, absolute_import
import tensorflow as tf
import os
import pandas as pd

TRAINING_FILE = 'adult.data'
EVAL_FILE = 'adult.test'
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'census_data')

CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

NUM_COLS = ['capital_gain', 'capital_loss', 'hours_per_week', 'education_num']

CATE_COLS = ['gender', 'race', 'education', 'marital_status', 'relationship',
             'workclass', 'occupation', 'native_country']
EMDED_COLS = ['occupation']
CROSSED_COLS = ['education', 'occupation']
BUCKETED_COLS = 'age'
LABEL_COL = 'income_bracket'

# download(DATA_DIR)

train_df = pd.read_csv(os.path.join(DATA_DIR, TRAINING_FILE), names=CSV_COLUMNS)
eval_df = pd.read_csv(os.path.join(DATA_DIR, EVAL_FILE), names=CSV_COLUMNS)
all_df = train_df.append(eval_df, ignore_index=True)


def get_cate_cols_values(df):
    cate_cols_values = {}
    for cate_col in CATE_COLS:
        cate_cols_values[cate_col] = df[cate_col].value_counts().index.tolist()
    return cate_cols_values


cate_cols_values = get_cate_cols_values(all_df)


# for cate_col, cate_cols_value in cate_cols_values.items():
#     print(cate_col + ":")
#     print(cate_cols_value)

def train_input_fn(data_file, batch_size=128):
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop(LABEL_COL)
        classes = tf.cast(tf.equal(labels, '>50K'), tf.int32)
        return features, classes

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
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop(LABEL_COL)
        classes = tf.cast(tf.equal(labels, '>50K'), tf.int32)
        return features, classes

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def get_wide_columns():
    wide_feature_columns = []
    for cate_col, cate_cols_value in cate_cols_values.items():
        wide_feature_columns.append(
            tf.feature_column.categorical_column_with_vocabulary_list(cate_col, cate_cols_value))
    wide_feature_columns.append(tf.feature_column.crossed_column(CROSSED_COLS, hash_bucket_size=1000))
    bucketized_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(BUCKETED_COLS),
                                                            boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    wide_feature_columns.append(bucketized_column)
    CROSSED_COLS.append(bucketized_column)
    wide_feature_columns.append(
        tf.feature_column.crossed_column(CROSSED_COLS, hash_bucket_size=1000))
    return wide_feature_columns


def get_deep_columns():
    deep_feature_columns = []
    for num_cols in NUM_COLS:
        deep_feature_columns.append(tf.feature_column.numeric_column(num_cols))
    for cate_col, cate_cols_value in cate_cols_values.items():
        if cate_col not in EMDED_COLS:
            deep_feature_columns.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(cate_col, cate_cols_value)))
        else:
            deep_feature_columns.append(tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(cate_col, cate_cols_value), dimension=8))
    return deep_feature_columns


def train_and_eval(model_type, train_path, eval_path, save_model_path):
    if model_type == 'wide':
        model = tf.estimator.LinearClassifier(feature_columns=get_wide_columns(), model_dir=save_model_path,
                                              optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                                                               l1_regularization_strength=0.0,
                                                                               l2_regularization_strength=10.0))
    elif model_type == 'deep':
        model = tf.estimator.DNNClassifier(feature_columns=get_deep_columns(), model_dir=save_model_path,
                                           hidden_units=[100, 75, 50, 25])
    else:
        model = tf.estimator.DNNLinearCombinedClassifier(linear_feature_columns=get_wide_columns(),
                                                         dnn_feature_columns=get_deep_columns(),
                                                         model_dir=save_model_path, dnn_hidden_units=[100, 75, 50, 25]
                                                         )
    model.train(input_fn=lambda: train_input_fn(train_path), max_steps=10000)
    model.evaluate(input_fn=lambda: eval_input_fn(eval_path))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_path = os.path.join(DATA_DIR, TRAINING_FILE)
    eval_path = os.path.join(DATA_DIR, EVAL_FILE)
    train_and_eval('wide_and_deep', train_path, eval_path, "wide_and_deep_model_with_estimators")


if __name__ == '__main__':
    tf.app.run()
