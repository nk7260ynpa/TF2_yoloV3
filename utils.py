import os
import numpy as np
from PIL import ImageDraw, Image
import tensorflow as tf

#模型輸出對anchor做sigmoid後處理
def post_process(model_output, img_size, anchors, num_classes):
    num_anchors = len(anchors)
    shape = model_output.shape.as_list()
    grid_size = shape[1:3]
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes
    predictions = tf.reshape(model_output, [-1, num_anchors * dim, bbox_attrs])
    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
    
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
    
    box_centers = tf.nn.sigmoid(box_centers)
    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride
    
    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride
    
    confidence = tf.nn.sigmoid(confidence)
    
    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    
    classes = tf.nn.sigmoid(classes)
    
    predictions = tf.concat([detections, classes], axis=-1)
    return predictions

#後處理座標轉換成min max
def center_to_minmax(detections):
    center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2
    
    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)
    return detections

def iou_calculate(box1, box2):
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2
    
    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)
    
    if (int_x0 > int_x1) or (int_y0 > int_y1):
        return 0.
    
    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
    
    box1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    box2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)
    
    iou = int_area / (box1_area + box2_area - int_area + 1e-5)
    return iou
    

    
def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    # 去除confidence過低的參數
    conf_mask = np.expand_dims((predictions_with_boxes.numpy()[:,:,4] > confidence_threshold), -1)
    predictions = predictions_with_boxes.numpy() * conf_mask
    
    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        print("shape1", shape)
        non_zero_idx = np.nonzero(image_pred)
        image_pred = image_pred[np.unique(non_zero_idx[0], axis=0)]
        print("shape2", image_pred.shape)
        image_pred = image_pred.reshape(-1, shape[-1])
        
        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
        
        unique_classes = list(set(classes.reshape(-1)))
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:,-1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]
            
            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([iou_calculate(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
                
                
    return result
        
def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))

def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def draw_boxes(boxes, img, cls_names, detections_size):
    draw = ImageDraw.Draw(img)
    
    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detections_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], "{} {:.2f}%".format(cls_names[cls], score * 100), fill=color)
            print("{} {:.2f}%".format(cls_names[cls], score * 100), box[:2])