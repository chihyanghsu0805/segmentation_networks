from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

class LearningRateScheduler(tf.keras.callbacks.Callback):
  """Learning rate scheduler which sets the learning rate according to schedule.
  Arguments:
      schedule: a function that takes an epoch index (integer, indexed from 0) and current learning rate as inputs and returns a new learning rate as output (float).
  """

  def __init__(self, schedule_list):
    super(LearningRateScheduler, self).__init__()
    self.schedule_list = schedule_list

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.lr_schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('\nEpoch %05d: Learning rate is %6.6f.' % (epoch, scheduled_lr))

  def lr_schedule(self, epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < int(self.schedule_list[0][0]) or epoch > int(self.schedule_list[-1][0]):
      return lr
    for i in range(len(self.schedule_list)):
      if epoch == int(self.schedule_list[i][0]):
        return float(self.schedule_list[i][1])
    return lr

