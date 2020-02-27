from __future__ import absolute_import
from __future__ import print_function

from  data_io import read_nifti_image

import math
import numpy as np
import tensorflow as tf

class TrainSequence(tf.keras.utils.Sequence):
  ''' Every Sequence must implement the __getitem__ and the __len__ methods.
      If you want to modify your dataset between epochs you may implement on_epoch_end.
      The method __getitem__ should return a complete batch.
  '''
  def __init__(self, inputs, targets, batch_size):
    self.x = inputs
    self.y = targets
    self.batch_size = batch_size

  def __len__(self):
    return math.ceil(len(self.x) / self.batch_size)

  def __getitem__(self, idx):
    #inputs = get_inputs(self.x, idx, self.batch_size)
    # Inputs
    batch_x = self.x[idx*self.batch_size : (idx+1)*self.batch_size]
    inputs = []
    for x in batch_x:
      cat_input = read_nii_from_list(x)
      inputs.append(cat_input)

    targets = get_targets(self.y, idx, self.batch_size)

    return np.array(inputs), targets

class DevSequence():
  def __init__(self, inputs, targets):
    self.x = inputs
    self.y = targets
    self.batch_size = len(inputs)
  
  def convert_to_numpy(self):
    #inputs = get_inputs(self.x, 0, self.batch_size)
    inputs = []
    for x in self.x:
      cat_input = read_nii_from_list(x)
      inputs.append(cat_input)

    self.inputs = np.array(inputs)

    targets = get_targets(self.y, 0, self.batch_size)

    self.targets = [np.array(x) for x in targets]

#def get_inputs(input_list, idx, batch_size):
#    inputs = []
#    batch = input_list[idx*batch_size : (idx+1)*batch_size] # Batch of sample
#    for x in batch: # for each sample do
#      input = read_nii_from_list(x) # Read list of sequence, output cat_image
#      inputs.append(input)
  
#    return inputs

def get_targets(target_list, idx, batch_size):
  targets = []

  for target in target_list:
    """ 2019/12/24 Implement Labels Target """
    temp_targets = []
    batch = target[idx*batch_size : (idx+1)*batch_size]
    for y in batch:
      if target[0][0].endswith('nii.gz'):
        _target = read_nii_from_list(y)
        _target = np.squeeze(_target)
      else: # Labels
        _target = int(y[0])
      temp_targets.append(_target)
    targets.append(temp_targets)

  return targets

def get_targets_old(target_list, idx, batch_size):
  targets = []
  num_targets = len(target_list)
  for target in target_list:
    temp_targets = []
    batch = target[idx*batch_size : (idx+1)*batch_size]
    for y in batch:
      target = read_nii_from_list(y)
      target = np.squeeze(target)
      temp_targets.append(target)
    targets.append(temp_targets)

  return targets

def read_nii_from_list(sequence_list):
  for idx, ith_sequence in enumerate(sequence_list):
    image = read_nifti_image(ith_sequence)
    image = np.expand_dims(image, axis=-1)
    if idx == 0:
      cat_image = image
    else:
      cat_image = np.concatenate((cat_image, image), axis=-1)

  return cat_image
