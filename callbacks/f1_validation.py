from __future__ import absolute_import
from __future__ import print_function

from sklearn.metrics import f1_score

import csv

import numpy as np
import tensorflow as tf

class F1Validation(tf.keras.callbacks.Callback):
  def __init__(self, dev_set, output_path, target_idx):
    super().__init__()
    #self.train_set = train_set
    self.dev_set = dev_set
    self.history = []
    self.output_path = output_path
    self.target_idx = target_idx

  def on_train_end(self, logs=None):
    with open(self.output_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerows(self.history)

    print('F1 history saved at:', self.output_path)

  def on_epoch_end(self, epoch, logs=None):
    true_input = self.dev_set[0]
    true_target = self.dev_set[1]

    num_samples = len(true_input)
    for sample in range(num_samples):
      input = np.expand_dims(true_input[sample], axis=0)
      pred_logits = self.model.predict(input, verbose=0)

      pred_logits = pred_logits[self.target_idx]
      true_labels = true_target[self.target_idx][sample]

      pred_labels = np.argmax(pred_logits, axis=-1)
      pred_labels.tolist()

      if sample == 0:
        F1 = f1_score(np.squeeze(true_labels.astype(int)).flatten(), pred_labels.flatten(), average=None)
      else:
        F1 = F1+f1_score(np.squeeze(true_labels.astype(int)).flatten(), pred_labels.flatten(), average=None)

    F1 = F1 / (num_samples)
    print('Average Validation F1 is ', ["{0:0.2f}".format(i) for i in F1])
    self.history.append(F1)
  def on_predict_batch_end(self, batch, logs=None):
    print('Predict Batch End')
  def on_predict_end(self, logs=None):
    print('Predict End')
