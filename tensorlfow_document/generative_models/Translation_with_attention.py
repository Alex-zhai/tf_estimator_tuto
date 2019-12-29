# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/21 10:21

import os
import re

import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers

tf.enable_eager_execution()

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"


def preprocess_sentence(w):
    w = w.lower().strip()

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples):
    lines = open(path, 'r').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return word_pairs


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above
    inp_lang = LanguageIndex(sp for en, sp in pairs)
    targ_lang = LanguageIndex(en for en, sp in pairs)

    # Vectorize the input and target languages

    # Spanish sentences
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]

    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length_inp,
                                                                 padding='post')

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,
                                                                  padding='post')

    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file,
                                                                                                 num_examples)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)


def train_input_fn(batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(count=1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


def eval_input_fn(batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.enc_units, return_state=True, return_sequences=True, recurrent_activation='sigmoid',
                              recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.dec_units, return_state=True, return_sequences=True, recurrent_activation='sigmoid',
                              recurrent_initializer='glorot_uniform')
        self.fc = layers.Dense(vocab_size)

        # used for attention
        self.W1 = layers.Dense(self.dec_units)
        self.W2 = layers.Dense(self.dec_units)
        self.V = layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # hidden_with_time_axis shape (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, axis=1)
        # enc_output shape: (batch_size, max_length, hidden_size)
        # score shape: (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)
        # (batch_size, max_length, 1) * (batch_size, max_length, hidden_size) = (batch_size, max_length, hidden_size)
        context_vector = attention_weights * enc_output
        #  (batch_size, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, shape=(-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


encoder = Encoder(vocab_size=len(inp_lang.word2idx), embedding_dim=256, enc_units=1024, batch_sz=64)
decoder = Decoder(vocab_size=len(targ_lang.word2idx), embedding_dim=256, dec_units=1024, batch_sz=64)


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask)


def train(batch_size=64, epochs=10):
    optimizer = tf.train.AdamOptimizer()
    train_dataset = train_input_fn(batch_size)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        hidden = encoder.initialize_hidden_state()
        for (batch, (inp, targ)) in enumerate(train_dataset):
            loss = 0
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * batch_size, axis=1)
                for t in range(1, targ.shape[1]):
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    dec_input = tf.expand_dims(targ[:, t], axis=1)
                    loss += loss_function(real=targ[:, t], pred=predictions)
            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / int((len(input_tensor_train) / batch_size))))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))


def eval(sentence):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word2idx[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros(shape=(1, 1024))]
    enc_output, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], axis=0)
    for t in range(max_length_targ):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.idx2word[predicted_id] + ' '
        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence


if __name__ == '__main__':
    train(batch_size=64, epochs=5)
    result, sentence = eval(u'hace mucho frio aqui.')
    print(result)
    print(sentence)
