# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/18 9:24

from __future__ import print_function, division, absolute_import
import tensorflow as tf
import urllib
# import urllib.request
# from urllib2 import urlopen
import os
import pandas as pd

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)
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
CROSSED_COLS = ['education', 'occupation']
BUCKETED_COLS = 'age'
LABEL_COL = 'income_bracket'


def download_and_clean_file(filename, url):
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
        with tf.gfile.Open(filename, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(", ", ",")
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[-1]
                line += '\n'
                eval_file.write(line)
    tf.gfile.Remove(temp_file)


def download(data_dir):
    tf.gfile.MakeDirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.gfile.Exists(training_file_path):
        download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.gfile.Exists(eval_file_path):
        download_and_clean_file(eval_file_path, EVAL_URL)


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


def get_feature_columns():
    feature_columns = []
    for num_col in NUM_COLS:
        feature_columns.append(tf.feature_column.numeric_column(num_col))
    for cate_col, cate_cols_value in cate_cols_values.items():
        feature_columns.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(cate_col, cate_cols_value)))
    # feature_columns.append(tf.feature_column.crossed_column(CROSSED_COLS, hash_bucket_size=1000))
    # bucketized_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(BUCKETED_COLS),
    #                                                         boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    # feature_columns.append(bucketized_column)
    # CROSSED_COLS.append(bucketized_column)
    # feature_columns.append(
    #     tf.feature_column.crossed_column(CROSSED_COLS, hash_bucket_size=1000))
    return feature_columns


def train_and_eval(train_path, eval_path, save_model_path):
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)
    config = tf.estimator.RunConfig(save_checkpoints_steps=1000, train_distribute=distribution)
    model = tf.estimator.DNNClassifier(hidden_units=[256, 128, 64], feature_columns=get_feature_columns(),
                                       model_dir=save_model_path, config=config,
                                       optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                                                        l1_regularization_strength=0.0,
                                                                        l2_regularization_strength=10.0))
    model.train(input_fn=lambda: train_input_fn(train_path), max_steps=10000)
    model.evaluate(input_fn=lambda: eval_input_fn(eval_path))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_path = os.path.join(DATA_DIR, TRAINING_FILE)
    eval_path = os.path.join(DATA_DIR, EVAL_FILE)
    train_and_eval(train_path, eval_path, "linear_model_with_estimators")


if __name__ == '__main__':
    tf.app.run()
