import os
import tensorflow as tf
from gpus import gpu_growth
from models import yolov3
from data_loader.VOC import VOC_dataset
from yolo_v3_loss import yolo_loss

gpu_growth("12")

IMAGE_PATH = "data/PascalVOC/VOC2012"
LABEL_PATH = "train_data/"
COCO_ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
image_size = 416

Dataset = VOC_dataset(IMAGE_PATH, LABEL_PATH)

yolov3_model = yolov3.Darknet_yolo(num_class=20)

yolov3_model.load_weights("weights/yolov3.h5",by_name=True, skip_mismatch=True)

optimizer = tf.optimizers.Adam(0.0001)

@tf.function
def train_step(yolov3_model, data, yolo_loss):
    with tf.GradientTape() as tape:
        loss = yolo_loss(yolov3_model, data[0], data[3], data[2], data[1], COCO_ANCHORS,image_size)
    gradients = tape.gradient(loss, yolov3_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, yolov3_model.trainable_variables))
    return loss

i = 0
for data in Dataset:
    loss = train_step(yolov3_model, data, yolo_loss)
    i += 1
    if i % 10 == 0:  
        print(i)
        print(loss.numpy())
        yolov3_model.save_weights("weights/VOC.h5")
    if i == 1000000:
        break