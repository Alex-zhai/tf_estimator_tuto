# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/19 10:44

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.utils import get_file

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'ModelYear', 'Origin']
feature_cols = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'ModelYear', 'Origin']

dataset_path = get_file("auto-mpg.data",
                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
raw_df = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

train_df = raw_df.sample(frac=0.8, random_state=1234)
test_df = raw_df.drop(train_df.index)
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

y_train = train_df.pop('MPG').values
y_test = test_df.pop("MPG").values
X_train = train_df.values
X_test = test_df.values

train_features_np_list = np.split(X_train, X_train.shape[1], axis=1)
eval_features_np_list = np.split(X_test, X_train.shape[1], axis=1)


def get_bucket_boundaries(feature):
    return np.unique(np.percentile(feature, range(0, 100))).tolist()


def train_input_fn():
    features = {feature_name: tf.constant(train_features_np_list[i]) for i, feature_name in enumerate(feature_cols)}
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensors(features), tf.data.Dataset.from_tensors(y_train)))
    return dataset


def eval_input_fn():
    features = {feature_name: tf.constant(eval_features_np_list[i]) for i, feature_name in enumerate(feature_cols)}
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensors(features), tf.data.Dataset.from_tensors(y_test)))
    return dataset


def get_feature_columns():
    feature_columns = []
    for i, col_name in enumerate(feature_cols):
        feature_columns.append(tf.feature_column.bucketized_column(tf.feature_column.numeric_column(col_name),
                                                                   get_bucket_boundaries(train_features_np_list[i])))
    return feature_columns


def train_and_eval(save_model_path):
    classifier = tf.contrib.estimator.boosted_trees_regressor_train_in_memory(train_input_fn,
                                                                              feature_columns=get_feature_columns(),
                                                                              model_dir=save_model_path, n_trees=50,
                                                                              max_depth=5)
    eval_results = classifier.evaluate(eval_input_fn)
    print(eval_results)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("boosted_trees_estimators")


if __name__ == '__main__':
    tf.app.run()
