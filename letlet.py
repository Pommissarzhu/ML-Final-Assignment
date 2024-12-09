import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LetVaeGru(tf.keras.Model):
    def __init__(self, time_step=75, feature_dim=66, latent_sequence=128):
        super().__init__()

        self.time_step = time_step
        self.feature_dim = feature_dim
        self.latent_sequence = latent_sequence

        # encoder
        self.lstm_1 = LSTM(units=64, return_sequences=True, activation='tanh')
        self.lstm_2 = LSTM(units=32, return_sequences=True, activation='tanh')
        self.lstm_3 = LSTM(units=16, return_sequences=False, activation='tanh')
        self.z_mean = Dense(units=self.latent_sequence)
        self.z_log_var = Dense(units=self.latent_sequence)

        # decoder
        self.dense1 = Dense(units=self.time_step * 16, activation='relu')
        self.reshape = Reshape((self.time_step, 16))
        self.lstm_4 = LSTM(units=16, return_sequences=True, activation='tanh')
        self.lstm_5 = LSTM(units=32, return_sequences=True, activation='tanh')
        self.lstm_6 = LSTM(units=64, return_sequences=True, activation='tanh')
        # self.lstm_7 = LSTM(units=self.feature_dim, return_sequences=True, activation='tanh')
        self.dense2 = Dense(units=self.feature_dim, activation='relu')

    def reparameterize(self, mean, log_var):
        # 从标准正态分布中采样 epsilon
        epsilon = tf.random.normal(shape=tf.shape(mean))
        # 计算 sigma（标准差）
        sigma = tf.exp(0.5 * log_var)
        # 计算重参数化后的潜在向量 z
        z = mean + sigma * epsilon
        return z

    def encode(self, x):
        # x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

    def decode(self, z_mean, z_log_var):
        z = self.reparameterize(z_mean, z_log_var)
        z = self.dense1(z)
        z = self.reshape(z)
        z = self.lstm_4(z)
        z = self.lstm_5(z)
        # z = self.lstm_6(z)
        # z = self.lstm_7(z)
        z = self.dense2(z)
        return z

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = self.encode(inputs)
        outputs = self.decode(z_mean, z_log_var)
        return outputs, z_mean, z_log_var


class DataLoader:
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
    # 计算逐元素的均方误差
    mse_loss = tf.keras.losses.mean_squared_error(x, x_pred)  # 输出形状: [batch_size, time_step, feature_dim]

    # 对时间步长和特征维度求和，保留 batch 维度
    reconstruction_loss = tf.reduce_sum(mse_loss, axis=-1)  # 输出形状: [batch_size]

    # 对 batch 取平均，得到整体重构损失
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)  # 输出标量

    # KL 散度损失
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)

    # 总的 VAE 损失
    vae_loss = reconstruction_loss + kl_loss
    return vae_loss

def test():
    model_to_be_restored = LetVaeGru(time_step, feature_dim, latent_sequence)
    checkpoint = tf.train.Checkpoint(model=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))
    data_loader = DataLoader()
    data = data_loader.get_batch(1)
    x_logit, _, _ = model_to_be_restored.call(data)
    plt.subplot(121)
    plt.plot(data[0])
    plt.subplot(122)
    plt.plot(x_logit[0])
    plt.title('Original vs Reconstructed')
    plt.legend()
    plt.savefig('r.png')
    plt.close()


def train():
    tf.keras.backend.clear_session()

    model = LetVaeGru(time_step, feature_dim, latent_sequence)

    data_loader = DataLoader()

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    log_dir = 'let_tensorboard_logs'
    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(profiler=True)

    num_batches = int(data_loader.num_data // batch_size * num_epochs)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    for batch_index in range(num_batches):
        data = data_loader.get_batch(batch_size)

        with tf.GradientTape() as tape:
            x_pred, mean, log_var = model(data)
            loss = compute_loss(data, x_pred, mean, log_var)
            # loss = tf.reduce_mean(loss)

            print("batch:", batch_index, "loss:", loss.numpy())

            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=batch_index)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if batch_index % 100 == 0:
            checkpoint.save(f'./save/{batch_index}_letModel.ckpt')


time_step = 75
feature_dim = 66
latent_sequence = 16

num_epochs = 10000
batch_size = 10
learning_rate = 1e-4

if __name__ == "__main__":
    train()
    # test()