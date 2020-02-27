from __future__ import absolute_import
from __future__ import print_function

#from models import *

from model_builder import ModelBuilder
from model_tester import ModelTester
from model_trainer import ModelTrainer

import argparse
import configparser
import csv
import os
import time

import tensorflow as tf

def get_dataset(config):
    # Input List
    with open(config['inputs'], "r") as csv_file:
      reader = csv.reader(csv_file)
      x_list = list(reader)

    # Target List
    target_list = config['targets'].split(',')
    y_list = []
    for i in target_list:
      with open(i, "r") as csv_file:
        reader = csv.reader(csv_file)
        ith_target_list = list(reader)
      y_list.append(ith_target_list) 

    return x_list, y_list

if __name__ == '__main__':
# If called as the main function and not a subfunction

  """ Parse Arguments
  model stores the path for model configurations.
  train stores the path for training configurations.
  test stores the path for testing configurations.
  gpu stores the gpu id used.
  """
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', dest='model', type=str, required=True)
  parser.add_argument('--train', dest='train', type=str)
  parser.add_argument('--test', dest='test', type=str)
  args = parser.parse_args()

  # Build Model Graph from Config  
  model_config = configparser.ConfigParser()
  model_config.read(args.model)
  model = ModelBuilder(model_config)
  model.build_graph()
  model.compile()
  model.summary_txt()
  model.print_png()
  model.save_graph()

  # Train Model
  if args.train:
    train_config = configparser.ConfigParser()
    train_config.read(args.train)
    trainer = ModelTrainer(train_config)
    trainer.get_hyperparameters()
    trainer.get_train_set()
    trainer.get_dev_set()
    trainer.get_callbacks()
    train_history = model.train(trainer.fit_options)
    model.save_weights(trainer.outputs_config['weights'])
    trainer.write_outputs(train_history)

  # Test Model
  if args.test:
    test_config = configparser.ConfigParser()
    test_config.read(args.test)    
    tester = ModelTester(test_config)
    tester.get_weights()
    model.load_weights(tester.weights)
    tester.get_test_set()
    tester.get_header_list()
    tester.get_output_list()
    tester.predict_and_write(model)
