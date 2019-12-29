# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/1/28 17:43

import tensorflow as tf

HEADER = ['key', 'x', 'y', 'alpha', 'beta', 'target']
HEADER_DEFAULTS = [[0], [0.0], [0.0], ['NA'], ['NA'], ['NA']]
TARGET_LABELS = ['positive', 'negative']
NUM_COLS = ['x', 'y']
CATE_COLS = {'alpha': ['ax01', 'ax02'], 'beta': ['bx01', 'bx02']}
TARGET_NAME = 'target'


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(TARGET_LABELS))
    return table.lookup(label_string_tensor)


def train_input_fn(train_csv_name):
    def parse_csv(value):
        decode_csv = tf.decode_csv(value, record_defaults=HEADER_DEFAULTS)
        features = dict(zip(HEADER, decode_csv))
        features.pop('key')
        labels = features.pop(TARGET_NAME)
        return features, labels

    dataset = tf.data.TextLineDataset(train_csv_name)
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, parse_label_column(batch_y)


if __name__ == '__main__':
    batch_x, batch_y = train_input_fn("train-data.csv")
    sess = tf.Session()
    sess.run(tf.tables_initializer())
    print(sess.run([batch_x, batch_y]))
