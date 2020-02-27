from __future__ import absolute_import
from __future__ import print_function

from callbacks import *
from data_io import write_history
from sequences import TrainSequence, DevSequence

import csv
import tensorflow as tf

class ModelTrainer():
  def __init__(self, config):
    self.datasets_config = config['datasets']
    self.hps_config = config['hyperparameters']
    self.callbacks_config = config['callbacks']
    self.outputs_config = config['outputs']
    self.fit_options = {}
    self.fit_options['callbacks'] = []

  def get_hyperparameters(self):
    self.fit_options['epochs'] = int(self.hps_config['num_epochs'])
    self.fit_options['batch_size'] = int(self.hps_config['batch_size'])
    self.fit_options['verbose'] = int(self.hps_config['verbose'])
    
  def get_train_set(self):
    inputs = get_file_list(self.datasets_config['train_inputs']) 
    targets = get_file_list(self.datasets_config['train_targets'])

    train_set = TrainSequence(inputs[0], targets, self.fit_options['batch_size']) # Only supports one input.
    self.fit_options['x'] = train_set

    num_samples = len(inputs[0])
    print('Number of Training Samples: ', num_samples)

    num_sequences = len(inputs[0][0])
    print('Number of Input Sequences: ', num_sequences)

    num_targets = len(targets)
    print('Number of Targets: ', num_targets)

  def get_dev_set(self):
    inputs = get_file_list(self.datasets_config['dev_inputs'])
    targets = get_file_list(self.datasets_config['dev_targets'])

    dev_set = DevSequence(inputs[0], targets) # Currently only single input
    dev_set.convert_to_numpy()

    #print(dev_set.inputs.shape)
    #print(dev_set.targets[0].shape)
    #print(dev_set.targets[1].shape)

    self.fit_options['validation_data'] = (dev_set.inputs, dev_set.targets)
    num_samples = len(dev_set.inputs)
    print('Number of Validation Samples: ', num_samples)

  def get_callbacks(self):
    # Checkpoints
    # Early Stopping
    # Learning Rate
    with open(self.callbacks_config['learning_rate_schedule'], "r") as csv_file:
      reader = csv.reader(csv_file)
      schedule = list(reader)
   
    learning_rate_callback = LearningRateScheduler(schedule)
    self.fit_options['callbacks'].append(learning_rate_callback)

    # YOLO
    if 'yolo' in self.callbacks_config:
      yolo_target = int(self.callbacks_config['yolo_target'])
      YoloCallback = YoloValidation(dev_set=self.fit_options['validation_data'],
                                    output_path=self.callbacks_config['yolo_history'],
                                    target_idx=yolo_target)
      self.fit_options['callbacks'].append(YoloCallback)


    # F1
    if 'f1' in self.callbacks_config:
      f1_target = int(self.callbacks_config['f1_target'])
      F1Callback = F1Validation(dev_set=self.fit_options['validation_data'],
                                output_path=self.callbacks_config['f1_history'],
                                target_idx=f1_target)
      self.fit_options['callbacks'].append(F1Callback)
    
    # MSE
    if 'mse' in self.callbacks_config:
      mse_target = int(self.callbacks_config['mse_target'])
      MSECallback = MSEValidation(dev_set=self.fit_options['validation_data'],
                                  output_path=self.callbacks_config['mse_history'],
                                  target_idx=mse_target)
      self.fit_options['callbacks'].append(MSECallback)

  def write_outputs(self, history):
    write_history(history.history, self.outputs_config['history'])

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

