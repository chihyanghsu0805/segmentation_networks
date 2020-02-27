from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

def convert_predictions(pred, num_box_attrs, num_classes):
  # pred.shape[0] = 1 which is num_samples
  grid_x = pred.shape[1]
  grid_y = pred.shape[2]
  grid_z = pred.shape[3]

  # Meshgrid for offsets
  # !!! grid[x][y] == (y, x)
  grid = tf.meshgrid(tf.range(grid_x), tf.range(grid_y), tf.range(grid_z)) # 
  grid = tf.stack(grid, axis=-1) # (8,8,16,3)
  #grid = tf.expand_dims(grid, axis=3)  # [grid_x, grid_y, grid_z, 1, 3] (4,4,8,1,3)
  grid = tf.cast(grid, tf.float32)
  # grid: 0:7,0:7,0:15 (8,8,16,3)

  #pred = tf.reshape(pred, (-1, grid_x, grid_y, grid_z, num_box_attrs+num_classes)) # (8,8,16,n_box,15)
  box_xyz, box_whl, confidence, class_probs = tf.split(pred, (3,3,1,num_classes), axis=-1, name='pred_split') # xyz, whl, confidence, classes
  # box_xyz: (4,4,8,n_box,3), tx,ty,tz
  # box_whl: (4,4,8,n_box,3), tw,th,tl
  # confidence: (4,4,8,n_box,1)
  # class_probs: (4,4,8,n_box,num_classes)

  # Box_XYZ
  box_xyz = tf.sigmoid(box_xyz)
  box_xyz = box_xyz+grid
  #/tf.cast(grid_x*grid_y*grid_z, tf.float32)
  # box_xyz: 0:1
  # bx = s(tx)+cx, by = s(ty)+cy, bz = s(tz)+cz
  # xyz is relative to grid

  # Box_WHL
  box_whl = tf.exp(box_whl)#*anchor_boxes
  # whl is relative to image
  # bw = pw*e(tw), bh = ph*e(th), bl = pl*e(tl)

  confidence = tf.sigmoid(confidence) # IOU

  #class_probs = tf.sigmoid(class_probs)
  class_probs= tf.math.softmax(class_probs)

  return box_xyz, box_whl, confidence, class_probs

def get_best_iou(box_1, box_2):
  box_1 = tf.expand_dims(box_1, -2) # (1,128*3,1,6) # 6 = XYZ, WHL
  box_2 = tf.expand_dims(box_2, 0) # (1,15,6) # 6 = XYZ, WHL
  # new_shape: (..., N, (x0,y0,z0,x1,y1,z1))
  new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
  box_1 = tf.broadcast_to(box_1, new_shape, name='box_1') # (4,4,8,N_Boxes,?,6)
  box_2 = tf.broadcast_to(box_2, new_shape, name='box_2') # (4,4,8,N_Boxes,?,6)
  
  box_1_area = (box_1[..., 3]-box_1[..., 0]) * (box_1[..., 4]-box_1[..., 1]) * (box_1[..., 5]-box_1[..., 2])
  box_2_area = (box_2[..., 3]-box_2[..., 0]) * (box_2[..., 4]-box_2[..., 1]) * (box_2[..., 5]-box_2[..., 2])

  intersect_w = tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 0], box_2[..., 0])
  intersect_w = tf.maximum(intersect_w, 0)

  intersect_h = tf.minimum(box_1[..., 4], box_2[..., 4]) - tf.maximum(box_1[..., 1], box_2[..., 1])
  intersect_h = tf.maximum(intersect_h, 0)

  intersect_l = tf.minimum(box_1[..., 5], box_2[..., 5]) - tf.maximum(box_1[..., 2], box_2[..., 2])
  intersect_l = tf.maximum(intersect_h, 0)

  int_area = intersect_w*intersect_h*intersect_l

  union_area = box_1_area+box_2_area-int_area
  iou = int_area / union_area

  return iou

def get_xyz_loss(true_xyz, pred_xyz):
  return tf.reduce_sum(tf.square(true_xyz-pred_xyz), axis=-1)

def get_whl_loss(true_whl, pred_whl):
  return tf.reduce_sum(tf.square(true_whl-pred_whl), axis=-1)

def get_confidence_loss(true_confidence, pred_confidence):
  return tf.keras.losses.binary_crossentropy(true_confidence, pred_confidence)

def get_class_loss(true_class, pred_class):
  return tf.keras.losses.sparse_categorical_crossentropy(true_class, pred_class)

def yolo_v2_loss(num_classes, num_box_attrs, grid_shape):

  def get_yolo_loss(y_true, y_pred):

    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
  
    # Predictions
    pred_xyz, pred_whl, pred_confidence, pred_class = convert_predictions(y_pred, num_box_attrs, num_classes)

    # pred_xyz = (8,8,16,3) in grid dimension
    # pred_whl = (8,8,16,3) relative to image
    # pred_confidence = (8,8,16,1)
    # pred_class = (8,8,16,num_classes)

    # Truth
    true_xyz, true_whl, true_confidence, true_class = tf.split(y_true, (3, 3, 1, 1), axis=-1, name='Truth_Split') # Truth should be in same configuration as Predictions
    # true_xyz = (8,8,16,3) in voxel dimension
    # true_whl = (8,8,16,3) in voxel dimension 
    # true_confidence = (8,8,16,1)
    # true_class = (8,8,16,1)

    true_xyz = true_xyz / 16 # Grid Dimension
    true_whl = true_whl / [128,128,256] # 0~1, relative to image
    
    # Box
    #true_box = tf.concat((true_xyz-true_whl/2, true_xyz+true_whl/2), axis=-1)
    #pred_box = tf.concat((pred_xyz-pred_whl/2, pred_xyz+pred_whl/2), axis=-1)
    true_whl_half = true_whl / 2.
    true_min = true_xyz - true_whl_half
    true_max = true_xyz + true_whl_half  

    pred_whl_half = pred_whl / 2.
    pred_min = pred_xyz - pred_whl_half
    pred_max = pred_xyz + pred_whl_half       

    # Batch Size
    batch_size = tf.shape(pred_confidence)[0]

    # 
    intersect_min = tf.maximum(pred_min,  true_min)
    intersect_max = tf.minimum(pred_max, true_max)
    intersect_whl = tf.maximum(intersect_max - intersect_min, 0.)
    intersect_areas = intersect_whl[..., 0] * intersect_whl[..., 1] * intersect_whl[..., 2]

    true_areas = true_whl[..., 0] * true_whl[..., 1] * true_whl[..., 2]
    pred_areas = pred_whl[..., 0] * pred_whl[..., 1] * pred_whl[..., 2]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    true_box_conf = iou_scores * true_confidence

    # Masks
    obj_mask = true_confidence
    noobj_mask = 1-true_confidence

    # Losses
    # Class Loss
    true_class = tf.squeeze(true_class)
    true_class = tf.cast(true_class, 'uint8')
    true_class_onehot = tf.one_hot(true_class, depth=8)
    diff = true_class_onehot-pred_class
    sq_diff = tf.math.square(diff)
    masked_diff = tf.math.multiply(obj_mask, sq_diff)
    class_loss = tf.reduce_sum(masked_diff)

    loss = class_loss


    return loss
  return get_yolo_loss

