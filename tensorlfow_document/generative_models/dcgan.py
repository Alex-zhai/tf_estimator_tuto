# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/2/22 14:53

from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.datasets import mnist

tf.enable_eager_execution()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = (x_train - 127.5) / 127.5


def train_input_fn(batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


def generator_model(input_shape=(100,)):
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(7 * 7 * 256, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((7, 7, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    out = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    return Model(inputs=inp, outputs=out)


def discriminator_model(input_shape=(28, 28, 1)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inp)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    return Model(inputs=inp, outputs=out)


def generator_loss(gen_out):
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(gen_out), logits=gen_out)


def discriminator_loss(gen_out, real_out):
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_out), logits=real_out)
    gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(gen_out), logits=gen_out)
    return real_loss + gen_loss


generator = generator_model()
discriminator = discriminator_model()


def train(batch_size=32, epoches=50):
    gen_optimizer = tf.train.AdamOptimizer(1e-4)
    disc_optimizer = tf.train.AdamOptimizer(1e-4)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    train_dataset = train_input_fn(batch_size)
    random_vector_for_generation = tf.random_normal([10, 100])
    for epoch in range(epoches):
        start = time.time()
        for batch, real_img in enumerate(train_dataset):
            noise = tf.random_normal([batch_size, 100])
            with tf.GradientTape as gen_tape, tf.GradientTape as disc_tape:
                gen_images = generator(noise)
                real_out = discriminator(real_img)
                gen_out = discriminator(gen_images)
                gen_loss = generator_loss(gen_out)
                disc_loss = discriminator_loss(gen_out, real_out)
                print("epoch {} batch {} gen loss is {}".format(epoch, batch, gen_loss))
                print("epoch {} batch {} disc loss is {}".format(epoch, batch, disc_loss))
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

            gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

        generate_and_save_images(generator, epoch, random_vector_for_generation)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    generate_and_save_images(generator, epoches, random_vector_for_generation)


def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        img_arr = predictions[i, :, :, 0] * 127.5 + 127.5
        img = Image.fromarray(img_arr)
        img.save('image_at_epoch_{:04d}_index_{:04d}.png'.format(epoch, i + 1))


if __name__ == '__main__':
    dataset = train_input_fn()
    for img in dataset:
        print(img.shape)
    train(batch_size=32, epoches=50)
