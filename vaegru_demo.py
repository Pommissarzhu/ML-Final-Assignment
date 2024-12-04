import tensorflow as tf
from tensorflow.keras import layers, Model

class VAEGRU(tf.keras.Model):
    def __init__(self, sequence_length, feature_dim, latent_dim):
        super(VAEGRU, self).__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # 编码器部分
        self.encoder_gru_1 = layers.GRU(64, return_sequences=True, name="gru_1")
        self.encoder_gru_2 = layers.GRU(32, name="gru_2")
        self.z_mean_layer = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_layer = layers.Dense(latent_dim, name="z_log_var")

        # 解码器部分
        self.repeat_vector = layers.RepeatVector(sequence_length, name="repeat_vector")
        self.decoder_gru_1 = layers.GRU(32, return_sequences=True, name="decoder_gru_1")
        self.decoder_gru_2 = layers.GRU(64, return_sequences=True, name="decoder_gru_2")
        self.decoder_output = layers.TimeDistributed(layers.Dense(feature_dim), name="decoder_output")

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def encode(self, inputs):
        x = self.encoder_gru_1(inputs)
        x = self.encoder_gru_2(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        x = self.repeat_vector(z)
        x = self.decoder_gru_1(x)
        x = self.decoder_gru_2(x)
        return self.decoder_output(x)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        outputs = self.decode(z)
        self.add_loss(self.compute_loss(inputs, outputs, z_mean, z_log_var))
        return outputs

    def compute_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.mean_squared_error(inputs, outputs), axis=1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        return reconstruction_loss + kl_loss

# 超参数设置
sequence_length = 75
feature_dim = 66
latent_dim = 16

# 构建模型
vae_gru = VAEGRU(sequence_length, feature_dim, latent_dim)
vae_gru.compile(optimizer=tf.keras.optimizers.Adam())

# 打印模型结构
inputs = tf.keras.Input(shape=(sequence_length, feature_dim))
vae_gru(inputs)
vae_gru.summary()
