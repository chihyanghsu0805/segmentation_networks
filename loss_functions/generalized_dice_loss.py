from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

def generalized_dice_loss(**kwargs):
  def loss(y_true, y_pred):

    eps = 1e-6
    y_true = tf.cast(y_true, tf.int64)
    y_one_hot = tf.one_hot(indices=y_true, depth=tf.shape(y_pred)[-1], dtype="float32")
    y_one_hot = tf.reshape(y_one_hot, tf.shape(y_pred))
  
    ref_vol = tf.reduce_sum(y_one_hot, axis=(0,1,2,3))
    weights = tf.reciprocal(tf.square(ref_vol)+1)  
  
    numerator = 2. * tf.reduce_sum(y_pred*y_one_hot, axis=(0,1,2,3))
    w_numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(y_pred, axis=(0,1,2,3)) + tf.reduce_sum(y_one_hot, axis=(0,1,2,3)) # Pred + RP
    w_denominator = tf.reduce_sum(weights*denominator)
   
    dsc = (w_numerator + eps) / (w_denominator + eps) # eps in both num/den => dsc=1 when class missing.
    cost = 1. - dsc
  
    return cost
  return loss

