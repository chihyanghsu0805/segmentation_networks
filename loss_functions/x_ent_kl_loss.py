from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
#from tensorflow.keras.losses import mse, binary_crossentropy

def x_ent_kl_loss(z_log_var, z_mean):

  def get_kl_loss():
    #z_log_var = loss_layers['z_log_var']
    #z_mean = loss_layers['z_mean']
    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return kl_loss

  def get_recon_loss(y_true, y_pred):
    #return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(loss_layers['input'], y_pred)

  def loss(y_true, y_pred):
    orig_dim = y_pred.get_shape().as_list()
    recon_loss = get_recon_loss(y_true, y_pred)
    recon_loss = np.prod(orig_dim[1:4])*recon_loss
    kl_loss = get_kl_loss()
    vae_loss = tf.keras.backend.mean(recon_loss + kl_loss)
    return vae_loss

  return loss

