from __future__ import absolute_import
from __future__ import print_function

from callbacks import *
from data_io import write_nifti_image, read_nifti_header
from sequences import read_nii_from_list

import csv
import numpy as np
import os
import tensorflow as tf

class ModelTester():
  """ 
    callbacks not available in model.predict() due to Keras version.
  """

  def __init__(self, config):
    self.datasets_config = config['datasets']
    self.model_config = config['model']
    self.outputs_config = config['outputs']
    self.predict_options = {}
    #self.predict_options['callbacks'] = []
    self.header_list = []

  def get_weights(self):    
    self.weights = self.model_config['weights']

  def get_test_set(self):
    inputs = get_file_list(self.datasets_config['test_inputs']) 
    self.test_set = inputs[0]

    num_samples = len(inputs[0])
    print('Number of Testing Samples: ', num_samples)

    num_sequences = len(inputs[0][0])
    print('Number of Input Sequences: ', num_sequences)

  def get_header_list(self):
    for sequence_list in self.test_set:
      first_sequence = sequence_list[0]
      header = read_nifti_header(first_sequence)
      self.header_list.append(header)

  def get_output_list(self):
    if 'logits' in self.outputs_config:
      self.logits = get_file_list(self.outputs_config['logits'])

    if 'labels' in self.outputs_config:
      self.labels = get_file_list(self.outputs_config['labels'])

  def predict_and_write(self, model):
    for sample_idx, (sample, header) in enumerate(zip(self.test_set, self.header_list)):
      inputs = read_nii_from_list(sample)
      inputs = np.expand_dims(inputs, axis=0)

      logits = model.model.predict(inputs, **self.predict_options)
      num_targets = len(logits)
      for target_idx, target in enumerate(logits):
        target = np.squeeze(target)
        if 'logits' in self.outputs_config:        
          output_path = self.logits[0][sample_idx][target_idx]
          save_dir = os.path.dirname(output_path)
          if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

          if output_path.endswith('.nii.gz'):
            #write_nifti_image(output_path, logits, header)

            write_nifti_image(output_path, target, header)
          #if output_path.endswith('.csv'):
      
        if 'labels' in self.outputs_config:
          output_path = self.labels[0][sample_idx][target_idx]
          save_dir = os.path.dirname(output_path)
          if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

          labels = np.argmax(target, axis=-1)
          labels = np.float32(labels)
          write_nifti_image(output_path, labels, header)

def get_file_list(csv_list):
  """ Read CSV from list.
  Args:
    csv_list: list of CSVs pointing to inputs/targets.
  Return: 
    file_list: List[List[List]] containing num_inputs/targets, num_samples, num_sequences.
  """

  csv_list = csv_list.split(',')
  file_list = []
  for csv_path in csv_list:
    with open(csv_path, "r") as csv_file:
      reader = csv.reader(csv_file)
      temp_list = list(reader)
    file_list.append(temp_list)

  return file_list

