import os
import sys
import time
import random
import string
import argparse
import numpy as np
import tensorflow as tf
from model import yolov3
from utils import detections_center_to_minmax, non_max_suppression
from utils import convert_to_original_size, load_coco_names, draw_boxes
from PIL import ImageDraw, Image

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='image for prediction', type=str)
parser.add_argument('--output', required=True, help='result of prediction', type=str)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# load image
img = Image.open(opt.input)
img_resized = img.resize(size=(416, 416))
img_resized = np.array(img_resized, dtype=np.float32)
img_resized = tf.expand_dims(tf.constant(img_resized)/255., axis=0)

# Build Model
yolov3_model = yolov3.model()
yolov3.load_weights("weights/yolov3.weights", yolov3_model)

model_output_1, model_output_2, model_output_3 = yolov3_model(img_resized)

ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (159, 198), (373, 326)]

predictions_1 = yolov3.detection_layer(model_output_1, (416, 416), ANCHORS[6:9], num_classes=80)
predictions_2 = yolov3.detection_layer(model_output_2, (416, 416), ANCHORS[3:6], num_classes=80)
predictions_3 = yolov3.detection_layer(model_output_3, (416, 416), ANCHORS[0:3], num_classes=80)

detections = tf.concat([predictions_1, predictions_2, predictions_3], axis=1)

boxes = detections_center_to_minmax(detections)

conf_threshold = 0.5
iou_threshold = 0.4
class_names = "coco.names"

filtered_boxes = non_max_suppression(boxes, confidence_threshold=conf_threshold, iou_threshold=iou_threshold)

classes = load_coco_names(class_names)

draw_boxes(filtered_boxes, img, classes, (416, 416))

img.save(opt.output)


