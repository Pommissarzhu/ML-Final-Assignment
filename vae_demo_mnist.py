# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:58:57 2019

@author: dyliang
"""

from __future__ import absolute_import, print_function, division
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import time
import plot


# vae
class vae(keras.Model):

    def __init__(self, latent_dim):
        super(vae, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.latent_dim + self.latent_dim)
        ])
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(units=7 * 7 * 32, activation='relu'),
            keras.layers.Reshape(target_shape=(7, 7, 32)),
            keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                strides=(1, 1),
                padding="SAME"),
            keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                strides=(1, 1),
                padding="SAME",
                activation='sigmoid'),
        ])

    def encode(self, x):
        mean_logvar = self.encoder(x)
        N = mean_logvar.shape[0]
        mean = tf.slice(mean_logvar, [0, 0], [N, self.latent_dim])
        logvar = tf.slice(mean_logvar, [0, self.latent_dim], [N, self.latent_dim])
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean


# training
class train:

    @staticmethod
    def compute_loss(model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logits = model.decode(z)

        # loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)
        marginal_likelihood = - tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)

        KL_divergence = tf.reduce_sum(mean ** 2 + tf.exp(logvar) - logvar - 1, axis=1)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence
        loss = -ELBO
        return loss

    @staticmethod
    def compute_gradient(model, x):
        with tf.GradientTape() as tape:
            loss = train.compute_loss(model, x)
        gradient = tape.gradient(loss, model.trainable_variables)
        return gradient, loss

    @staticmethod
    def update(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))


# hpy
latent_dim = 100
num_epochs = 100
lr = 1e-4
batch_size = 1000
train_buf = 60000
test_buf = 10000


# load data
def load_data(batch_size):
    mnist = keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32') / 255.
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32') / 255.

    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size)
    train_dataset = tf.data.Dataset.zip((train_data, train_labels)).shuffle(train_buf)

    test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)
    test_labels = tf.data.Dataset.from_tensor_slices(test_labels).batch(batch_size)
    test_dataset = tf.data.Dataset.zip((test_data, test_labels)).shuffle(test_buf)

    return train_dataset, test_dataset


# begin training
def begin():
    train_dataset, test_dataset = load_data(batch_size)
    model = vae(latent_dim)
    optimizer = keras.optimizers.Adam(lr)

    for epoch in range(num_epochs):
        start = time.time()
        last_loss = 0
        for train_x, _ in train_dataset:
            gradients, loss = train.compute_gradient(model, train_x)
            train.update(optimizer, gradients, model.trainable_variables)
            last_loss = loss
        # if epoch % 10 == 0:
        print('Epoch {},loss: {},Remaining Time at This Epoch:{:.2f}'.format(
            epoch, last_loss, time.time() - start))

    plot.plot_VAE(model, test_dataset)


if __name__ == '__main__':
    begin()

