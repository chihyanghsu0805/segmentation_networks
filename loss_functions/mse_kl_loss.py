from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
#from tensorflow.keras.losses import mse, binary_crossentropy
import numpy as np

def mse_kl_loss(z_log_var, z_mean):

  def get_kl_loss():
    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return kl_loss

  def get_reconstruction_loss(y_true, y_pred):
    return tf.keras.losses.mse(y_true, y_pred)

  def loss(y_true, y_pred):
    original_dim = y_pred.get_shape().as_list()
    original_dim.pop(0)
    reconstruction_loss = get_reconstruction_loss(y_true, y_pred)
    #reconstruction_loss = np.prod(original_dim)*reconstruction_loss
    kl_loss = get_kl_loss()
    kl_loss = kl_loss/np.prod(original_dim)
    vae_loss = reconstruction_loss+kl_loss
    return vae_loss

  return loss

