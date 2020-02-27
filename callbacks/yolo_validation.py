from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

#from yolo_convert_targets import get_bounding_box_with_class
from sklearn.metrics import log_loss

def get_best_iou(box_1, box_2):
  box_1 = np.expand_dims(box_1, -2) # (1,128,N_Boxes,1,6) # 6 = XYZ, WHL
  box_2 = np.expand_dims(box_2, 0) # (1,?,6) # 6 = XYZ, WHL

  box_1_area = (box_1[..., 3]-box_1[..., 0]) * (box_1[..., 4]-box_1[..., 1]) * (box_1[..., 5]-box_1[..., 2])
  box_2_area = (box_2[..., 3]-box_2[..., 0]) * (box_2[..., 4]-box_2[..., 1]) * (box_2[..., 5]-box_2[..., 2])

  intersect_w = np.maximum(np.minimum(box_1[..., 3], box_2[..., 3]) - np.maximum(box_1[..., 0], box_2[..., 0]), 0)
  intersect_h = np.maximum(np.minimum(box_1[..., 4], box_2[..., 4]) - np.maximum(box_1[..., 1], box_2[..., 1]), 0)
  intersect_l = np.maximum(np.minimum(box_1[..., 5], box_2[..., 5]) - np.maximum(box_1[..., 2], box_2[..., 2]), 0)
  int_area = intersect_w*intersect_h*intersect_l

  union_area = box_1_area+box_2_area-int_area
  iou = int_area / union_area

  return iou

def sigmoid(x):    
    return 1/(1+np.exp(-x))

def convert_predictions(pred, anchor_boxes, n_box_attrs, n_classes):
  n_anchor_boxes = len(anchor_boxes)
  grid_x = pred.shape[1]
  grid_y = pred.shape[2]
  grid_z = pred.shape[3]

  grid = np.meshgrid(np.arange(grid_x), np.arange(grid_y), np.arange(grid_z)) # [(4,4,8), (4,4,8), (4,4,8)]
  grid = np.stack(grid, axis=-1) # (4,4,8,3)
  grid = np.expand_dims(grid, axis=3)  # [grid_x, grid_y, grid_z, 1, 3] (4,4,8,1,3)
  grid = grid.astype(np.float32)
  pred = np.reshape(pred, (-1, grid_x, grid_y, grid_z, n_anchor_boxes, n_box_attrs+n_classes)) # (4,4,8,n_box,14)
  box_xyz, box_whl, confidence, class_probs = np.split(pred, [3,6,7], axis=-1) # xyz, whl, confidence, classes

  box_xyz = sigmoid(box_xyz)
  box_xyz = box_xyz+grid
  
  box_whl = np.exp(box_whl)*anchor_boxes
  confidence = sigmoid(confidence)
  class_probs = sigmoid(class_probs)

  return box_xyz, box_whl, confidence, class_probs

class YoloValidation(tf.keras.callbacks.Callback):
  def __init__(self, train_set, **kwargs):
    super().__init__()
    self.train_set = train_set
    self.graph = kwargs['graph']
    if 'dev_set' in kwargs:
      self.dev_set = kwargs['dev_set']

    self.history = []
    self.output_path = kwargs['output_path']

  def on_epoch_end(self, epoch, logs=None):
    true_input = self.dev_set[0]
    true_target = self.dev_set[1]
    n_cases = len(self.dev_set[0])
    d_list = []
    for x in self.graph:
      d = {}
      d['anchor_boxes'] = [x['anchor_boxes'][y] for y in x['anchor_masks']]
      d['n_box_attrs'] = x['n_box_attrs']
      d['n_classes'] = x['n_classes']
      d_list.append(d)    

    #for ith_case in range(n_cases):
    for ith_case in range(1):
      input = np.expand_dims(true_input[ith_case], axis=0)
      pred_logits = self.model.predict(input, verbose=0) # [(4,4,8,n_boxes*(xyzwhl+confidence+n_classes)), (8,8,16,n_boxes*(xyzwhl+confidence+n_classes)), ...]
      
      for i, (ith_logit, ith_dict, ith_truth) in enumerate(zip(pred_logits, d_list, true_target)):
        #ith_logit = pred_logits[0]
        #ith_truth = true_target[0]

        ith_truth = ith_truth.astype(np.float32)
        ith_logit = ith_logit.astype(np.float32)
        # Truth
        true_xyz, true_whl, true_confidence, true_class = np.split(ith_truth, (3, 6, 7), axis=-1) # Truth should be in same configuration as Predictions
        true_class = true_class.astype(int)

        ### Predictions
        pred_xyz, pred_whl, pred_confidence, pred_class_probs = convert_predictions(ith_logit, **ith_dict)# box_xyz, box_whl, confidence, class_probs     
        pred_xyz = pred_xyz*self.graph[i]['n_grids']
        pred_class = np.argmax(pred_class_probs, axis=-1)

        # Box
        true_box = np.concatenate((true_xyz-true_whl/2, true_xyz+true_whl/2), axis=-1) # (4,4,8,n_box,6)
        pred_box = np.concatenate((pred_xyz-pred_whl/2, pred_xyz+pred_whl/2), axis=-1) # (4,4,8,n_box,6)
        
        # Confidence
        mask_confidence = np.squeeze(true_confidence, -1)
        mask_confidence = mask_confidence.astype(bool)

        true_box_mask = true_box[mask_confidence]
        # tf version bug
        pred_box_flat = np.reshape(pred_box, (pred_box.shape[0], pred_box.shape[1]*pred_box.shape[2]*pred_box.shape[3]*pred_box.shape[4], 6))
        best_iou = get_best_iou(pred_box_flat, true_box_mask)
        best_iou = np.reshape(best_iou, (pred_box.shape[0], pred_box.shape[1], pred_box.shape[2], pred_box.shape[3], pred_box.shape[4], -1))
        # tf version bug
        
        a = np.reshape(true_class[mask_confidence], (1,-1))
        print(a[0])
        print(pred_class[mask_confidence])

        #print(true_box_mask) # (1,6)
        #print(pred_box[mask_confidence]) # (1,128,1,6)

        #print(pred_confidence[mask_confidence])

        best_iou = np.max(best_iou, axis=-1) # (4,4,8,n_boxes)
        bool_ignore = best_iou < 0.3

        box_loss_scale = 2-true_whl[...,0]*true_whl[...,1]*true_whl[...,2]/(128*128*256)

        xyz_loss = np.sum(np.square(true_xyz-pred_xyz), axis=-1)
        xyz_loss = xyz_loss*box_loss_scale
        xyz_loss = mask_confidence*xyz_loss
        xyz_loss = np.sum(xyz_loss)

        whl_loss =  np.sum(np.square(true_whl-pred_whl), axis=-1)
        whl_loss = whl_loss*box_loss_scale
        whl_loss = mask_confidence*whl_loss
        whl_loss = np.sum(whl_loss)

        true_confidence = np.reshape(true_confidence, (-1,1))
        pred_confidence = np.reshape(pred_confidence, (-1,1))

        confidence_loss = true_confidence*np.log(pred_confidence+1e-6)
        confidence_loss = confidence_loss+(1-true_confidence)*np.log(1-pred_confidence+1e-6)
        confidence_loss = -confidence_loss

        obj_loss = np.reshape(mask_confidence, (-1,1))*confidence_loss
        noobj_loss = np.reshape((1-mask_confidence), (-1,1))*np.reshape(bool_ignore, (-1,1))*confidence_loss

        confidence_loss = obj_loss+noobj_loss
        confidence_loss = np.sum(confidence_loss)

        true_class = true_class.astype(int)
        one_hot_targets = np.eye(8)[np.reshape(true_class, (-1))]
        #print(np.reshape(pred_class_probs[mask_confidence], (-1,8)))

        class_loss = log_loss(one_hot_targets, np.reshape(pred_class_probs, (-1,8)))
        class_loss = mask_confidence*class_loss
        class_loss = np.sum(class_loss)
        print("%1.2f, %1.2f, %1.2f, %1.2f, %1.2f" % (xyz_loss,whl_loss,np.sum(obj_loss),np.sum(noobj_loss),1000*class_loss))

  def on_train_end(self, logs=None):
    true_input = self.dev_set[0]
    true_target = self.dev_set[1]
    n_cases = len(self.dev_set[0])
    d_list = []
    for x in self.graph:
      d = {}
      d['anchor_boxes'] = [x['anchor_boxes'][y] for y in x['anchor_masks']]
      d['n_box_attrs'] = x['n_box_attrs']
      d['n_classes'] = x['n_classes']
      d_list.append(d)

    #for ith_case in range(n_cases):
    for ith_case in range(1):
      input = np.expand_dims(true_input[ith_case], axis=0)
      pred_logits = self.model.predict(input, verbose=0) # [(4,4,8,n_boxes*(xyzwhl+confidence+n_classes)), (8,8,16,n_boxes*(xyzwhl+confidence+n_classes)), ...]

      for i, (ith_logit, ith_dict, ith_truth) in enumerate(zip(pred_logits, d_list, true_target)):
        ith_truth = ith_truth.astype(np.float32)
        ith_logit = ith_logit.astype(np.float32)
        
        true_box_conf_class = get_truth(ith_truth)
        print(true_box_conf_class)
        i_pred_box_conf_class = get_pred(ith_logit, self.graph[i]['n_grids'], ith_dict)
        if i == 0:
          pred_box_conf_class = i_pred_box_conf_class
        else:
          pred_box_conf_class = np.concatenate([pred_box_conf_class, i_pred_box_conf_class], axis=0)
  
      score_threshold = 0.3
      score_mask = pred_box_conf_class[:,-2] > score_threshold

      pred_box_conf_class = pred_box_conf_class[score_mask]

      best_box_conf_class = yolo_non_max_suppression(pred_box_conf_class)
      print(best_box_conf_class)

def get_pred(pred, n_grids, d):
  pred_xyz, pred_whl, pred_confidence, pred_class_probs = convert_predictions(pred, **d)# box_xyz, box_whl, confidence, class_probs
  pred_xyz = pred_xyz*n_grids
  pred_class = np.argmax(pred_class_probs, axis=-1)
  pred_class = np.expand_dims(pred_class, axis=-1)
  pred_boxes = np.concatenate([pred_xyz-pred_whl/2,
                               pred_xyz+pred_whl/2], axis=-1)

  pred_box_conf_class = np.concatenate([pred_boxes, pred_confidence, pred_class], axis=-1)
  pred_box_conf_class = np.reshape(pred_box_conf_class, [-1, d['n_box_attrs']+1])
  return pred_box_conf_class

def get_truth(truth):
  ### Truth
  true_xyz, true_whl, true_confidence, true_class = np.split(truth, (3, 6, 7), axis=-1) # Truth should be in same configuration as Predictions
  true_box = np.concatenate([true_xyz-true_whl/2,
                             true_xyz+true_whl/2], axis=-1)

  # Confidence
  mask_confidence = np.squeeze(true_confidence, -1) == 1
  # mask_confidence: (4,4,8,n_box)
  true_xyz = true_xyz[mask_confidence]
  true_box_mask = true_box[mask_confidence] # ignore confidence = 0 and flatten
  true_box_class = np.concatenate((true_box_mask, true_class[mask_confidence]), axis=-1)
  return true_box_class

def yolo_non_max_suppression(all_property):
  best_boxes = []

  for i_class in range(8):
    class_mask = (all_property[:,7] == i_class)
    class_boxes = all_property[class_mask]
    if class_boxes.any():      
      while len(class_boxes) > 0:
        max_ind = np.argmax(class_boxes[:,6]) # Box w/ highest confidence
        best_box = class_boxes[max_ind]
        best_boxes.append(best_box)

        class_boxes = np.concatenate([class_boxes[: max_ind], class_boxes[max_ind+1:]]) # All other boxes
        iou = bboxes_iou(best_box[np.newaxis, :6], class_boxes[:, :6]) # Compute IOU w/ Best Box
        weight = np.ones((len(iou),), dtype=np.float32)

        iou_mask = iou > 0.5 # High Overlap
        weight[iou_mask] = 0.0 # Remove

        class_boxes[:, 6] = class_boxes[:, 6] * weight
        score_mask = class_boxes[:, 6] > 0.
        class_bboxes = class_boxes[score_mask]

  return best_boxes
      
def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 3] - boxes1[..., 0]) * (boxes1[..., 4] - boxes1[..., 1]) * (boxes1[..., 5] - boxes1[..., 2])
    boxes2_area = (boxes2[..., 3] - boxes2[..., 0]) * (boxes2[..., 2] - boxes2[..., 1]) * (boxes2[..., 5] - boxes2[..., 2])

    left_up       = np.maximum(boxes1[..., :3], boxes2[..., :3])
    right_down    = np.minimum(boxes1[..., 3:], boxes2[..., 3:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

