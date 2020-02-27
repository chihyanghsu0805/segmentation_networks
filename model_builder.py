from __future__ import absolute_import
from __future__ import print_function

from loss_functions import *
from models import *

import os
import tensorflow as tf
import time

class ModelBuilder():
  def __init__(self, config):
    self.graph, self.graph_config = parse_graph(config['graph'])
    self.compile_config = parse_compile(config['compile'])
    self.summary_config = config['summary']
    self.print_config = config['print']
    self.save_config = config['save']

    # Make Save Directory
    save_dir = os.path.dirname(config['save']['filepath'])
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

  def build_graph(self):
    self.model = self.graph.build(self.graph, **self.graph_config)

  def compile(self):
    """ read optimizer and loss from config and compile model.
    """
    options = {}   

    # Optimizer
    options['optimizer'] = self.compile_config['optimizer']
    
    # Loss Functions
    custom_loss = self.compile_config.pop('custom_loss')
    custom_loss = custom_loss.split(',')
    losses = self.compile_config['loss'].split(',')
    loss_list = []
    for loss, bool_custom in zip(losses, custom_loss):
      if bool_custom == 'True':
        loss_list.append(globals()[loss](**self.graph.loss_options))
      else:
        loss_list.append(loss)      

    options['loss'] = loss_list
 
    # Loss Weights
    if 'loss_weights' in self.compile_config:
      loss_weights = self.compile_config['loss_weights'].split(',')
      loss_weights = list(map(float, loss_weights))
      options['loss_weights'] = loss_weights
    
    self.model.compile(**options)

  def summary_txt(self):
    #self.model.summary()
    with open(self.summary_config['to_file'], 'w') as fh:
      # Pass the file handle in as a lambda function to make it callable
      self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
    print('Summary printed at: ', self.summary_config['to_file'])
  
  def print_png(self):
    tf.keras.utils.plot_model(self.model, **self.print_config)
    print('Model printed at: ', self.print_config['to_file'])

  def save_graph(self):
    self.model.save(**self.save_config)
    print('Model saved at: ',  self.save_config['filepath'])

  def train(self, fit_options):
    device = '/device:GPU:%s' % 1
    with tf.device(device):
      start_time = time.time()
      train_history = self.model.fit(**fit_options)
      end_time = time.time()
      print('Time elapsed: {}'.format(end_time - start_time))
    
    return train_history

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)

def parse_compile(config):
  return config

def parse_graph(config):
  type = config['type']
  graph = globals()[type]
  graph_config = graph.parse_config(config)
  return graph, graph_config
   



