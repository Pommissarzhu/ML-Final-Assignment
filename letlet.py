import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
from tensorflow.python.ops.gen_summary_ops import summary_writer


class LetVaeGru(tf.keras.Model):
    def __init__(self, time_step=75, feature_dim=66, latent_sequence=128):
        super().__init__()

        self.time_step = time_step
        self.feature_dim = feature_dim
        self.latent_sequence = latent_sequence

        # encoder
        self.gru_1 = GRU(units=64, return_sequences=True, activation='tanh')
        self.gru_2 = GRU(units=32, return_sequences=True, activation='tanh')
        self.gru_3 = GRU(units=16, return_sequences=True, activation='tanh')
        self.flatten = Flatten()
        self.z_mean = Dense(self.latent_sequence)
        self.z_log_var = Dense(self.latent_sequence)

        # decoder
        self.dense1 = Dense(units=75 * 16, activation='relu')
        self.reshape = Reshape((75, 16))
        self.gru_4 = GRU(units=32, return_sequences=True, activation='tanh')
        self.gru_5 = GRU(units=64, return_sequences=True, activation='tanh')
        self.gru_6 = GRU(units=66, return_sequences=True, activation='tanh')

    def reparameterize(self, mean, log_var):
        # 从标准正态分布中采样 epsilon
        epsilon = tf.random.normal(shape=tf.shape(mean))
        # 计算 sigma（标准差）
        sigma = tf.exp(0.5 * log_var)
        # 计算重参数化后的潜在向量 z
        z = mean + sigma * epsilon
        return z

    def encode(self, x):
        x = self.gru_1(x)
        x = self.gru_2(x)
        x = self.gru_3(x)
        x = self.flatten(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        # z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var

    def decode(self, z_mean, z_log_var):
        z = self.reparameterize(z_mean, z_log_var)
        z = self.dense1(z)
        z = self.reshape(z)
        z = self.gru_4(z)
        z = self.gru_5(z)
        z = self.gru_6(z)
        return z

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = self.encode(inputs)
        outputs = self.decode(z_mean, z_log_var)
        return outputs, z_mean, z_log_var

class DataLoader():
    def __init__(self):
        data_raw = pd.read_csv('data_m07_correct.csv', header=None)
        data_inc_raw = pd.read_csv('data_m07_incorrect.csv', header=None)
        data = np.array(data_raw).astype('float32')
        data_inc = np.array(data_inc_raw).astype('float32')
        self.data = data.reshape((-1, 75, 66))
        self.data_inc = data_inc.reshape((-1, 75, 66))
        self.num_data = data.shape[0]
    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.data)[0], batch_size)
        return self.data[index]

def compute_loss(x, x_pred, z_mean, z_log_var):
    # 1. 计算重建损失（均方误差）
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_pred), axis=[1, 2]))  # 按样本和特征维度计算MSE

    # 2. 计算KL散度
    # KL divergence between N(mu, sigma^2) and N(0, I)
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))

    # 3. 总损失 = 重建损失 + KL损失
    total_loss = reconstruction_loss + kl_loss
    return total_loss

@tf.function
def train_one_step(x):
    with tf.GradientTape() as tape:
        x_pred, mean, log_var = model(data)
        loss = compute_loss(data, x_pred, mean, log_var)
        loss = tf.reduce_mean(loss)
        tf.print("loss:", loss)
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=batch_index)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    time_step = 75
    feature_dim = 66
    latent_sequence = 128

    num_epochs = 1000
    batch_size = 10
    learning_rate = 1e-4

    model = LetVaeGru(time_step, feature_dim, latent_sequence)
    data_loader = DataLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    log_dir = 'let_tensorboard_logs'
    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(profiler=True, graph=True)

    num_batches = int(data_loader.num_data // batch_size * num_epochs)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    for batch_index in range(num_batches):
        data = data_loader.get_batch(batch_size)
        print(f'batch: {batch_index}')
        train_one_step(data)
        # with tf.GradientTape() as tape:
        #     x_pred, mean, log_var = model(data)
        #     loss = compute_loss(data, x_pred, mean, log_var)
        #     loss = tf.reduce_mean(loss)
        #     print("batch:", batch_index, "loss:", loss.numpy())
        #     with summary_writer.as_default():
        #         tf.summary.scalar("loss", loss, step=batch_index)
        # grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if batch_index % 100 == 0:
            checkpoint.save(f'./save/{batch_index}_letModel.ckpt')

