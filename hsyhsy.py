import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import pandas as pd
import time

class hsy_vae_gru(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def encode(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return z_mean, z_log_var

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def build_encoder(self):
        inputs = Input(shape=(75, 66))
        gru1 = GRU(units=64, return_sequences=True)(inputs)
        gru2 = GRU(units=32, return_sequences=True)(gru1)
        gru3 = GRU(units=16, return_sequences=True)(gru2)
        flatten = Flatten()(gru3)
        z_mean = Dense(self.latent_dim)(flatten)
        z_log_var = Dense(self.latent_dim)(flatten)
        z = self.reparameterize(z_mean, z_log_var)
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self):
        inputs = Input(shape=(self.latent_dim,))
        dense = Dense(75 * 16)(inputs)
        reshaped = Reshape((75, 16))(dense)
        gru1 = GRU(units=16, return_sequences=True)(reshaped)
        gru2 = GRU(units=32, return_sequences=True)(gru1)
        gru3 = GRU(units=64, return_sequences=True)(gru2)
        x = GRU(units=66, return_sequences=True)(gru3)
        return tf.keras.Model(inputs, x, name='decoder')

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=(tf.shape(mean)[0], mean.shape[1]))
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

class train:
    @staticmethod
    def compute_loss(model, x):
        mean, log_var = model.encode(x)
        z = model.reparameterize(mean, log_var)
        x_logit = model.decode(z)
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(x, x_logit)
        kl_divergence = tf.reduce_sum(mean ** 2 + tf.exp(log_var) - log_var - 1, axis=1)
        kl_divergence = tf.reduce_mean(kl_divergence)

        ELBO = reconstruction_loss - kl_divergence
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

latent_dim = 100
num_epochs = 1000
lr = 1e-4
batch_size = 1000
train_buf = 60000
test_buf = 10000

def data_loader():
    data_raw = pd.read_csv('data_m07_correct.csv', header=None)
    data_inc_raw = pd.read_csv('data_m07_incorrect.csv', header=None)
    data = np.array(data_raw).astype('float32')
    data_inc = np.array(data_inc_raw).astype('float32')
    data = data.reshape((-1, 75, 66))
    data_inc = data_inc.reshape((-1, 75, 66))
    return data, data_inc

def begin():
    data, data_inc = data_loader()
    model = hsy_vae_gru(latent_dim=100)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(num_epochs):
        start = time.time()
        last_loss = 0
        for item in data:
            item = np.expand_dims(item, axis=0)
            gradients, loss = train.compute_gradient(model, item)
            train.update(optimizer, gradients, model.trainable_variables)
            last_loss = loss
        print('Epoch {},loss: {},Remaining Time at This Epoch:{:.2f}'.format(
            epoch, last_loss, time.time() - start))

    # Save the model in H5 format
    model.save('hsy_vae_model', save_format='tf')

if __name__ == '__main__':
    begin()
