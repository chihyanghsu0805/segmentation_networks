from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

class VNetVAE():
  def __init__():
    pass

  def build(self, input_shape, num_levels, num_filters, kernel_size, padding, activation, pool_size, num_convs, dropout_rate, num_classes, latent_dimensions):
    X = tf.keras.Input(input_shape)
    inputs = X
  
    # Linear projection to match dimensions
    X = tf.keras.layers.Conv3D(filters=num_filters[0], kernel_size=1, padding='same', activation=None)(X)
    
    X, X_concat = vnet_encoder(X, num_levels, num_filters, kernel_size, padding, activation, pool_size, num_convs)
    encoder_endpoint = X  
    X = vnet_decoder(X, X_concat, num_levels, num_filters, kernel_size, padding, activation, pool_size, num_convs)

    X = tf.keras.layers.Dropout(dropout_rate)(X)
    vnet_output = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=1, strides=1, padding='same', activation='softmax', name='vnet')(X)

    # VD in paper
    decoder_x = tf.keras.layers.Conv3D(filters=num_filters[0], kernel_size=kernel_size[0], strides=pool_size[0], padding='same')(encoder_endpoint)
    decoder_flatten = tf.keras.layers.Flatten()(decoder_x)

    # Sampling
    z_mean = tf.keras.layers.Dense(latent_dimensions)(decoder_flatten)
    z_log_var = tf.keras.layers.Dense(latent_dimensions)(decoder_flatten)
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dimensions,), name='z')([z_mean, z_log_var])

    # Decoder settings
    endpoint_shape = decoder_x.get_shape().as_list()  
    endpoint_shape.pop(0)
    num_units = np.prod(endpoint_shape)
    num_channels = input_shape[-1]
 
    # VU in paper
    X = tf.keras.layers.Dense(units=num_units, activation='relu')(z)  
    X = tf.keras.layers.Reshape(target_shape=endpoint_shape)(X)
    X = tf.keras.layers.Conv3DTranspose(filters=num_filters[0], kernel_size=1, strides=pool_size[0], padding='same', activation='relu')(X)

    X = vnet_decoder(X, None, num_levels, num_filters, kernel_size, padding, activation, pool_size, num_convs)  
    vae_output = tf.keras.layers.Conv3D(filters=num_channels, kernel_size=1, padding='same', name='vae')(X)  
    
    outputs = [vnet_output, vae_output]
    model = tf.keras.Model(inputs, outputs)

    self.loss_options = {}
    self.loss_options['z_mean'] = z_mean
    self.loss_options['z_log_var'] = z_log_var

    return model

  def parse_config(config):
    """ Parse configurations from sting to int/float.
    Args:
      config: dict containing string
    Returns:
      graph: dict containing string/int/float 

    """
    graph = {}
    input_shape = config['input_shape'].split(',')
    graph['input_shape'] = tuple(map(int, input_shape))

    graph['num_levels'] = int(config['num_levels'])
    
    num_convs = config['num_convs'].split(",")
    graph['num_convs'] = tuple(map(int, num_convs))

    num_filters = config['num_filters'].split(",")
    graph['num_filters'] = tuple(map(int, num_filters))

    pool_size = config['pool_size'].split(",")
    graph['pool_size'] = tuple(map(int, pool_size))

    kernel_size = config['kernel_size'].split(";")
    kernel_size = [x.split(',') for x in kernel_size]
    graph['kernel_size'] = tuple([int(y) for y in x] for x in kernel_size)

    graph['padding'] = config['padding'].split(",")
    graph['activation'] = config['activation'].split(",")

    graph['num_classes'] = int(config['num_classes'])
    graph['dropout_rate'] = float(config['dropout_rate'])

    graph['latent_dimensions'] = int(config['latent_dimensions'])
    return graph

def vnet_encoder(X, num_levels, num_filters, kernel_size, padding, activation, pool_size, num_convs):
  X_concat = []
  # Encoder
  for i in range(num_levels):
    X_residual = X
    for j in range(num_convs[i]):
      X = tf.keras.layers.Conv3D(filters=num_filters[i], kernel_size=kernel_size[i], padding=padding[i], activation=activation[i])(X)
    X = tf.keras.layers.Add()([X, X_residual])

    # Pooling Replaced by Stride 2 Conv
    if i != num_levels-1:
      X_concat.append(X)
      X = tf.keras.layers.Conv3D(filters=num_filters[i+1], kernel_size=kernel_size[i], padding=padding[i], strides=pool_size[i], activation=activation[i])(X)

  return X, X_concat
  
def vnet_decoder(X, X_concat, num_levels, num_filters, kernel_size, padding, activation, pool_size, num_convs):
  # Decoder
  for i in range(num_levels-1, 0, -1):
    X = tf.keras.layers.Conv3DTranspose(filters=num_filters[i], kernel_size=kernel_size[i], padding=padding[i], strides=pool_size[i-1], activation=activation[i])(X)
    X_residual = X

    if X_concat:
      X = tf.keras.layers.Concatenate()([X, X_concat.pop()])

    for j in range(num_convs[i]):
      X = tf.keras.layers.Conv3D(filters=num_filters[i], kernel_size=kernel_size[i], padding=padding[i], activation=activation[i])(X)
  
    X = tf.keras.layers.Add()([X, X_residual])

  return X

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    Args:
      args (tensor): mean and log of variance of Q(z|X)

    Returns:
        z (tensor): sampled latent vector
    """
    
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

