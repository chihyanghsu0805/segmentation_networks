from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

class VNetYOLO():
  def __init__():
    pass

  def build(self, input_shape, num_levels, num_filters, kernel_size, padding, activation, pool_size, num_convs, dropout_rate, num_classes, num_box_attrs):
    """ Build vnet model from graph.
    Args:
      input_shape: shape of input with no batch. 
      num_levels: number of levels in VNet encoder. 
      num_filters: number of filters for convolution block. 
      kernel_size: size of kernel of each convolution block.
      padding: padding option for each convolution block.
      activation: activation option for each convolution block.
      pool_size: stride for pooling.
      num_convs: number of convlutions in each block.
      dropout_rate: rate for dropout.
      num_classes: number of output classes.
    
    Return:
      model: tf.keras.Model

    """
    X = tf.keras.Input(input_shape)
    inputs = X
    X_concat = []
    
    # Linear projection to match dimensions
    X = tf.keras.layers.Conv3D(filters=num_filters[0], kernel_size=1, padding='same', activation=None)(X)
    
    # Encoder
    for i in range(num_levels):
      X_residual = X
      for j in range(num_convs[i]):
        X = tf.keras.layers.Conv3D(filters=num_filters[i], kernel_size=kernel_size[i], padding=padding[i], activation=activation[i])(X)

      X = tf.keras.layers.Add()([X, X_residual])

      # Pooling Replaced by Stride 2 Conv
      if i != num_levels-1:
        X_concat.append(X)
        X = tf.keras.layers.Conv3D(filters=num_filters[i+1], kernel_size=kernel_size[i], padding=padding[i], strides=pool_size[i], activation=activation[i])(X)

    X_yolo = tf.keras.layers.Conv3D(filters=1024, kernel_size=3, strides=1, padding='same')(X)
    X_yolo = tf.keras.layers.Conv3D(filters=1024, kernel_size=3, strides=1, padding='same')(X_yolo)
    X_yolo = tf.keras.layers.Conv3D(filters=1024, kernel_size=3, strides=1, padding='same')(X_yolo)    
    X_yolo = tf.keras.layers.Conv3D(filters=num_classes+num_box_attrs, kernel_size=1, strides=1, padding='same', name='yolo')(X_yolo)

    # Decoder
    for i in range(num_levels-1, 0, -1):
      X = tf.keras.layers.Conv3DTranspose(filters=num_filters[i], kernel_size=kernel_size[i], padding=padding[i], strides=pool_size[i-1], activation=activation[i])(X)
      X_residual = X
      X = tf.keras.layers.Concatenate()([X, X_concat.pop()])

      for j in range(num_convs[i]):
        X = tf.keras.layers.Conv3D(filters=num_filters[i], kernel_size=kernel_size[i], padding=padding[i], activation=activation[i])(X)
  
      X = tf.keras.layers.Add()([X, X_residual])

    X = tf.keras.layers.Dropout(dropout_rate)(X)
    vnet_output = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=1, strides=1, padding='same', activation='softmax', name='vnet')(X)

    self.loss_options = {}
    self.loss_options['num_classes'] = num_classes
    self.loss_options['num_box_attrs'] = num_box_attrs
    self.loss_options['grid_shape'] = [8,8,16]
    
    outputs = [vnet_output, X_yolo]
    model = tf.keras.Model(inputs, outputs)
    return model

  def parse_config(config):
    """ Parse configurations from string to int/float.
    Args:
      config: dict containing string
    Returns:
      graph: dict containing string/int/float 

    """
    graph = {}
    input_shape = config['input_shape'].split(',')
    graph['input_shape'] = tuple(map(int, input_shape))

    graph['num_levels'] = int(config['num_levels'])
    
    num_convs = config['num_convs'].split(",")
    graph['num_convs'] = tuple(map(int, num_convs))

    num_filters = config['num_filters'].split(",")
    graph['num_filters'] = tuple(map(int, num_filters))

    pool_size = config['pool_size'].split(",")
    graph['pool_size'] = tuple(map(int, pool_size))

    kernel_size = config['kernel_size'].split(";")
    kernel_size = [x.split(',') for x in kernel_size]
    graph['kernel_size'] = tuple([int(y) for y in x] for x in kernel_size)

    graph['padding'] = config['padding'].split(",")
    graph['activation'] = config['activation'].split(",")

    graph['num_classes'] = int(config['num_classes'])
    graph['dropout_rate'] = float(config['dropout_rate'])

    graph['num_box_attrs'] = int(config['num_box_attrs'])
    return graph
