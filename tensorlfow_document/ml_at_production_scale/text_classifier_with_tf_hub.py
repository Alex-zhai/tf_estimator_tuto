# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/20 16:21

import tensorflow as tf
import os
import re
import pandas as pd
import tensorflow_hub as hub
from tensorflow.python.keras.utils import get_file


def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets():
    dataset = get_file(fname="aclImdb.tar.gz",
                       origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                       extract=True)
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
    return train_df, test_df


train_df, test_df = download_and_load_datasets()


def train_input_fn():
    return tf.estimator.inputs.pandas_input_fn(train_df, train_df['polarity'], num_epochs=None, shuffle=True)


def eval_input_fn():
    return tf.estimator.inputs.pandas_input_fn(test_df, test_df['polarity'], shuffle=False)


def get_hub_embedding():
    embedding_column = hub.text_embedding_column(key="sentence",
                                                 module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    return embedding_column


def train_and_eval(save_model_path):
    embedding_column = get_hub_embedding()
    model = tf.estimator.DNNClassifier(hidden_units=[500, 100], feature_columns=[embedding_column],
                                       model_dir=save_model_path, n_classes=2,
                                       optimizer=tf.train.AdamOptimizer(learning_rate=0.003))
    model.train(input_fn=lambda: train_input_fn(), max_steps=1000)
    model.evaluate(input_fn=lambda: eval_input_fn())


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("text_classifier_with_tf_hub")


if __name__ == '__main__':
    tf.app.run()
