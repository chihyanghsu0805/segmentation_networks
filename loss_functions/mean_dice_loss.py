from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

def mean_dice_loss(**kwargs):
  def loss(y_true, y_pred):

  #def dsc(p_y_given_x_train, y_gt, eps=1e-5):
  # Similar to Intersection-Over-Union / Jaccard above.
  # Dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    eps = 1e-6
    y_gt = tf.cast(y_true, tf.int64)
    p_y_given_x_train = y_pred 

    y_one_hot = tf.one_hot(indices=y_gt, depth=tf.shape(p_y_given_x_train)[-1], dtype="float32")
    y_one_hot = tf.reshape(y_one_hot, tf.shape(p_y_given_x_train))
    numer = 2. * tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,1,2,3)) # 2 * TP
    denom = tf.reduce_sum(p_y_given_x_train, axis=(0,1,2,3)) + tf.reduce_sum(y_one_hot, axis=(0,1,2,3)) # Pred + RP

    dsc = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
    av_class_dsc = tf.reduce_mean(dsc) # Along the class-axis. Mean DSC of classes. 
    cost = 1. - av_class_dsc

    return cost
  return loss
