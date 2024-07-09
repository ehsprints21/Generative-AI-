import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model





def build_vae(latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=(100, 2))  # Assuming M points are (100, 2)
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(256, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(latent_inputs)
    x = layers.Dense(100 * 2, activation='sigmoid')(x)  # Output shape matches M points
    decoder_outputs = layers.Reshape((100, 2))(x)

    decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    # VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, outputs, name='vae')

    # VAE loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(encoder_inputs, outputs)
    reconstruction_loss *= 100  # Adjust reconstruction loss weight

    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5

    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae

# Build VAE model
latent_dim = 32  # Adjust as needed
vae = build_vae(latent_dim)
vae.compile(optimizer='adam')
