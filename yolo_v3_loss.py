import os
import tensorflow as tf

ignore_thresh=0.5
grid_scale=1
obj_scale=5
noobj_scale=1
xywh_scale=1
class_scale=1
COCO_ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]

def _create_mesh_xy(batch_size, grid_h, grid_w, n_anchors):
    mesh_x = tf.cast(tf.reshape((tf.tile(tf.range(grid_w), [grid_h])), (1, grid_w, grid_h, 1, 1)), tf.float32)
    mesh_y = tf.transpose(mesh_x, (0, 2, 1, 3, 4))
    mesh_xy = tf.tile(tf.concat((mesh_x, mesh_y), -1), (batch_size, 1, 1, n_anchors, 1))
    return mesh_xy

def adjust_pred_tensor(y_pred):
    grid_offset = _create_mesh_xy(*y_pred.shape[:4])
    pred_xy = grid_offset * tf.sigmoid(y_pred[..., :2])
    pred_wh = y_pred[..., 2:4]
    pred_conf = tf.sigmoid(y_pred[..., 4:5])
    pred_classes = y_pred[..., 5:]
    pred = tf.concat([pred_xy, pred_wh, pred_conf, pred_classes], axis=-1)
    return pred

def _create_mesh_anchor(anchors, batch_size, grid_h, grid_w, n_anchors):
    mesh_anchor = tf.tile(anchors, [batch_size*grid_h*grid_w])
    mesh_anchor = tf.reshape(mesh_anchor, [batch_size, grid_h, grid_w, n_anchors, 2])
    mesh_anchor = tf.cast(mesh_anchor, tf.float32)
    return mesh_anchor

def conf_delta_tensor(y_true, y_pred, anchors, ignore_thresh, image_size):
    
    down_size = tf.math.divide(image_size, y_true.shape[1])
    pred_xy, pred_wh, pred_conf = y_pred[..., :2], y_pred[..., 2:4], y_pred[..., 4:5]
    true_xy, true_wh, true_conf = y_true[..., :2], y_true[..., 2:4], y_true[..., 4:5]
    
    
    #Transform to original locate
    true_xy = true_xy * down_size
    true_anchor_grid = _create_mesh_anchor(anchors, *y_true.shape[0:4])
    true_wh = true_anchor_grid * tf.exp(true_wh)
    true_wh = true_wh * true_conf
    true_wh_half = true_wh / 2.
    true_mins  = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    
    #Transform to original locate
    pred_xy = pred_xy * down_size
    pred_anchor_grid = _create_mesh_anchor(anchors, *y_pred.shape[0:4])
    pred_wh = pred_anchor_grid * tf.exp(pred_wh)
    pred_wh = pred_wh * pred_conf
    pred_wh_half = pred_wh / 2.
    pred_mins  = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half
    
    intersect_mins  = tf.math.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.math.minimum(pred_maxes, true_maxes)
    intersect_wh = intersect_maxes - intersect_mins 
    intersect_area = intersect_wh[..., 0:1] * intersect_wh[..., 1:2]
    
    true_area =  true_wh[..., 0:1] * true_wh[..., 1:]
    pred_area =  pred_wh[..., 0:1] * pred_wh[..., 1:]
    
    union_areas = true_area + pred_area - intersect_area 
    best_iou = tf.truediv(intersect_area, union_areas)
    
    conf_delta = pred_conf * tf.cast(best_iou < ignore_thresh, tf.float32)
    return conf_delta
    
def wh_scale_tensor(y_true, anchors, image_size):
    anchor_grid = _create_mesh_anchor(anchors, *y_true.shape[0:4])
    wh_scale = tf.exp(y_true[..., 2:4]) * anchor_grid / image_size
    wh_scale = (2 - wh_scale[...,0:1] * wh_scale[...,1:2])
    return wh_scale

def loss_coord_tensor(object_mask, y_true, y_pred, wh_scale, xywh_scale):
    xy_delta = object_mask * (y_true[..., 0:4] - y_pred[..., 0:4]) * wh_scale * xywh_scale
    loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
    return loss_xy

def loss_conf_tensor(object_mask, y_true, y_pred, obj_scale, noobj_scale, conf_delta):
    conf_delta = object_mask * (y_true[..., 4:5] - y_pred[..., 4:5]) * obj_scale + (1-object_mask) * conf_delta * noobj_scale
    loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))
    return loss_conf

def loss_class_tensor(object_mask, y_true, y_pred, class_scale):
    true_classes = tf.cast(y_true[..., 5:], tf.int64)
    pred_classes = tf.nn.softmax(y_pred[..., 5:])
    Sparse_loss = tf.losses.SparseCategoricalCrossentropy(reduction="none")
    class_delta = object_mask * tf.expand_dims(Sparse_loss(true_classes, pred_classes), axis=4)
    loss_class = tf.reduce_sum(class_delta, list(range(1, 5))) * class_scale
    return loss_class

def lossCalculator(y_true, y_pred, anchors, image_size):
    y_pred = tf.reshape(y_pred, (*y_true.shape[:4], -1))
    object_mask = y_true[..., 4:5]
    y_pred = adjust_pred_tensor(y_pred)
    
    
    conf_delta = conf_delta_tensor(y_true, y_pred, anchors, ignore_thresh, image_size)
    wh_scale = wh_scale_tensor(y_true, anchors, image_size)
    
    loss_xy = loss_coord_tensor(object_mask, y_true, y_pred, wh_scale, xywh_scale)
    loss_conf = loss_conf_tensor(object_mask, y_true, y_pred, obj_scale, noobj_scale, conf_delta)
    loss_class = loss_class_tensor(object_mask, y_true, y_pred, class_scale)
    loss = loss_xy + loss_conf + loss_class
    return loss * grid_scale

def yolo_loss(model, X_train, y_true_s, y_true_m, y_true_l, anchors,image_size):
    y_pred_s, y_pred_m, y_pred_l = model(X_train)
    loss_s = lossCalculator(y_true_s, y_pred_s, anchors[12:], image_size)
    loss_m = lossCalculator(y_true_m, y_pred_m, anchors[6:12], image_size)
    loss_l = lossCalculator(y_true_l, y_pred_l, anchors[:6], image_size)
    return tf.reduce_mean(tf.sqrt(loss_s+loss_m+loss_l))

