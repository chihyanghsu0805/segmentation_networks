from __future__ import absolute_import
from __future__ import print_function

import csv

import numpy as np
import tensorflow as tf

class MSEValidation(tf.keras.callbacks.Callback):
  def __init__(self, dev_set, output_path, target_idx):
    super().__init__()
    self.dev_set = dev_set
    self.history = []
    self.output_path = output_path
    self.target_idx = target_idx

  def on_train_end(self, logs=None):
    with open(self.output_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(self.history)

    print('MSE history saved at:', self.output_path)

  def on_epoch_end(self, epoch, logs=None):
    true_input = self.dev_set[0]
    true_target = self.dev_set[1]

    num_samples = len(true_input)
    for sample in range(num_samples):
      input = np.expand_dims(true_input[sample], axis=0)
      pred_logits = self.model.predict(input, verbose=0)

      pred_logits = pred_logits[self.target_idx]
      true_logits = true_target[self.target_idx][sample]

      if sample == 0:
        mse = np.mean(np.square(true_logits-pred_logits))
      else:
        mse = mse+np.mean(np.square(true_logits-pred_logits))
  
    mse = mse / num_samples
    print('Average Validation MSE is ', "{0:0.2f}".format(mse))
    self.history.append(mse)
