from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

class YoloValidation(tf.keras.callbacks.Callback):
  def __init__(self, dev_set, output_path, target_idx):
    super().__init__()
    #self.train_set = train_set
    self.dev_set = dev_set
    self.history = []
    self.output_path = output_path
    self.target_idx = target_idx

  def on_epoch_end(self, epoch, logs=None):
    true_input = self.dev_set[0]
    true_target = self.dev_set[1]

    num_samples = len(true_input)
    for sample in range(num_samples):
      input = np.expand_dims(true_input[sample], axis=0)
      pred_logits = self.model.predict(input, verbose=0)

      pred_yolo = pred_logits[self.target_idx] # (1,8,8,16,15)
      true_yolo = true_target[self.target_idx][sample] # (8,8,16,8)    

      true_confidence = true_yolo[...,6]
      true_confidence = true_confidence.astype(bool)
      true_class = true_yolo[...,7]
      true_class = true_class.astype(int)
      print(true_class[true_confidence])

      pred_class = np.argmax(pred_yolo[...,7:], axis=-1)
      pred_class = np.squeeze(pred_class)
      print(pred_class[true_confidence])





