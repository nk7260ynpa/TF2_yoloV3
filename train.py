import os
import tensorflow as tf
from gpus import gpu_growth
from models import yolov3
from data_loader.yolo import YOLO_dataset
from yolo_v3_loss import yolo_loss
import argparse



parser = argparse.ArgumentParser()
# System
parser.add_argument("--image_path", type=str, default="data/PascalVOC/VOC2012/JPEGImages", help="Image path")
parser.add_argument("--label_path", type=str, default="data/Transform_data", help="Label path")
parser.add_argument("--image_size", type=int, default=416, help="Specify the GPU number")
parser.add_argument("--anchor", help="Anchor size", type=str, required=False)

parser.add_argument("--model_type", type=str, default="Darknet53", choices=["Darknet53", "Efficientnet"], 
                    help="Darknet53 or Efficientnet")
parser.add_argument("--classes", type=int, default=20, help="classes")
parser.add_argument("--pretrain_weight", default="weights/Darknet53.h5",type=str, help="pretrained weight")
parser.add_argument("--output_weight", default="weights/output.h5", type=str, help="output weight")
parser.add_argument("--GPU", type=str, default="0", help="Specify the GPU number")


opt = parser.parse_args()

gpu_growth(op.GPU)

IMAGE_PATH = opt.image_path
LABEL_PATH = opt.label_path
IMAGE_SIZE = opt.image_size
CLASSES = opt.classes
PRETRAINED_WEIGHT = opt.pretrain_weight
OUTPUT_WEIGHT = opt.output_weight
MODEL_TYPE = opt.model_type

if opt.anchor == None:
    ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
else:
    ANCHORS = opt.anchor.split(",")

Dataset = YOLO_dataset(IMAGE_PATH, LABEL_PATH)

if MODEL_TYPE == "Darknet53": 
    yolov3_model = yolov3.Darknet_yolo(num_class=CLASSES)
else:
    yolov3_model = yolov3.Efficient_yolo(num_class=CLASSES)
    
if PRETRAINED_WEIGHT != None:
    yolov3_model.load_weights("weights/yolov3.h5",by_name=True, skip_mismatch=True)

optimizer = tf.optimizers.Adam(0.0001)

@tf.function
def train_step(yolov3_model, data, yolo_loss, IMAGE_SIZE):
    with tf.GradientTape() as tape:
        loss = yolo_loss(yolov3_model, data[0], data[3], data[2], data[1], ANCHORS, IMAGE_SIZE)
    gradients = tape.gradient(loss, yolov3_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, yolov3_model.trainable_variables))
    return loss

i = 0
for data in Dataset:
    loss = train_step(yolov3_model, data, yolo_loss, IMAGE_SIZE)
    i += 1
    if i % 10 == 0:  
        print(i)
        print(loss.numpy())
        yolov3_model.save_weights(OUTPUT_WEIGHT)
    if i == 1000000:
        break