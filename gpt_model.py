import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0., stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAEGRU:
    def __init__(self, time_steps=75, feature_dim=66, latent_dim=16, learning_rate=1e-4):
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()

    def build_encoder(self):
        inputs = Input(shape=(self.time_steps, self.feature_dim))
        x = GRU(32, return_sequences=False)(inputs)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,))
        x = Dense(self.time_steps * 32, activation='relu')(decoder_inputs)
        x = tf.reshape(x, (-1, self.time_steps, 32))
        decoder_gru = GRU(32, return_sequences=True)
        x = decoder_gru(x)
        outputs = Dense(self.feature_dim)(x)
        return Model(decoder_inputs, outputs, name='decoder')

    def build_vae(self):
        inputs = Input(shape=(self.time_steps, self.feature_dim))
        z_mean, z_log_var, z = self.encoder(inputs)
        vae_outputs = self.decoder(z)
        vae = Model(inputs, vae_outputs, name='vae')

        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(inputs, vae_outputs), axis=-1))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))
        vae_loss = reconstruction_loss + kl_loss
        vae.add_loss(vae_loss)

        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return vae

    def train(self, data, epochs=1000, batch_size=8, checkpoint_dir='checkpoints', log_dir='logs'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 设置TensorBoard日志记录
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, current_time)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        for epoch in range(epochs):
            start_time = time.time()
            history = self.vae.fit(data, epochs=1, batch_size=batch_size, verbose=0, callbacks=[tensorboard_callback])
            end_time = time.time()
            loss = history.history['loss'][0]
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f} seconds")

            # 每隔100个epoch保存一次checkpoint
            # if (epoch + 1) % 100 == 0:
            # checkpoint_path = os.path.join(checkpoint_dir, f'vae_checkpoint_epoch_{epoch + 1}')
            # self.vae.save(checkpoint_path)

    def save_model(self, save_dir='saved_model'):
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存整个VAE模型（使用h5格式）
        self.vae.save(os.path.join(save_dir, 'vae_5000.h5'), save_format='h5')
        # 保存编码器和解码器
        self.encoder.save(os.path.join(save_dir, 'encoder_5000.h5'), save_format='h5')
        self.decoder.save(os.path.join(save_dir, 'decoder_5000.h5'), save_format='h5')

    @staticmethod
    def load_and_infer(model_dir='saved_model', test_data=None, time_steps=75, feature_dim=66,
                       output_image_path='reconstruction.png'):
        # 加载模型（h5格式）
        loaded_vae = tf.keras.models.load_model(os.path.join(model_dir, 'vae_5000.h5'),
                                                custom_objects={'Sampling': Sampling})
        # 如果未提供测试数据，则生成随机数据
        if test_data is None:
            test_data = np.random.rand(1, time_steps, feature_dim).astype(np.float32)
        elif len(test_data.shape) == 2:
            # 如果提供的是二维数据，将其转换为三维数据
            test_data = np.expand_dims(test_data, axis=0)

        reconstructed_data = loaded_vae.predict(test_data)

        # 使用matplotlib显示结果
        # plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(test_data[0, :, :])
        plt.subplot(122)
        plt.plot(reconstructed_data[0, :, :])
        plt.title('Original vs Reconstructed')
        plt.legend()
        plt.savefig(output_image_path)
        plt.close()


def data_loader():
    data_raw = pd.read_csv('data_m07_correct.csv', header=None)
    data = np.array(data_raw).astype('float32')
    data = data.reshape((-1, 75, 66))
    return data


# 使用VAEGRU类
if __name__ == "__main__":
    # 数据维度
    TIME_STEPS = 75
    FEATURE_DIM = 66

    # 实例化VAE-GRU模型
    vae_gru = VAEGRU(time_steps=TIME_STEPS, feature_dim=FEATURE_DIM, latent_dim=64, learning_rate=1e-4)

    # 输入数据格式示例：
    # 数据应该是一个三维NumPy数组，形状为 (样本数, 时间步长, 特征维度)
    # 比如，数据集包含 70 个样本，每个样本有 75 个时间步长，每个时间步长包含 66 个特征
    # 生成一些虚拟数据用于测试（实际使用时请使用自己的数据集）
    # data = np.random.rand(70, TIME_STEPS, FEATURE_DIM).astype(np.float32)

    data = data_loader()

    # 训练模型
    # vae_gru.train(data, epochs=2500, batch_size=8)

    # 保存模型
    # vae_gru.save_model()

    # vae_gru.load_and_infer(model_dir='saved_model', test_data=data[0, :, :], time_steps=TIME_STEPS, feature_dim=FEATURE_DIM, output_image_path='r.png')

    # 加载模型并进行推理
    # 加载模型并进行推理并保存结果图像
    for i in range(data.shape[0]):
        VAEGRU.load_and_infer(model_dir='saved_model', test_data=data[i, :, :], time_steps=TIME_STEPS,
                              feature_dim=FEATURE_DIM, output_image_path=f'outputs/reconstruction_{i}.png')
