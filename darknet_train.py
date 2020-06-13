import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from imagenet_info import imagenet_info
from imagenet import imagenet_info, imagenet_train, imagenet_valid
from model.Darknet53_model import Darknet53
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "13"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

imgnet_info  = imagenet_info("data/imagenet/map_clsloc.txt", "data/imagenet/ILSVRC2012_validation_ground_truth.txt")
imgnet_train = imagenet_train("/raid/peterchen/datasets/imagenet/2012/train/", 256, imgnet_info, batch_size=128)
imgnet_valid = imagenet_valid("/raid/peterchen/datasets/imagenet/2012/valid/", 256, imgnet_info, batch_size=128)

model = Darknet53()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(0.0001)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
valid_loss = tf.keras.metrics.Mean(name="valid_loss")
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    
@tf.function
def valid_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)
    valid_loss(loss)
    valid_accuracy(labels, predictions)
    
time = datetime.datetime.now()
#weight_name = "weights"+'_'+str(time.month)+'-'+str(time.day)+'/'
weight_name = "weights.h5"
weights_path = os.path.join("weights", weight_name)
#if not os.path.exists(weights_path):
#    os.mkdir(weights_path)

EPOCHS = 50


best_loss = np.inf
img_num = 0
log = open(f'./train_record.txt', 'a')
print("Training Start!")
log.write("Training Start!\n")
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    
    count = 0
    for images, labels in imgnet_train.Dataset:
        train_step(images, labels)
        count += 1
        img_num += 128
        if count % 500 == 0:
            print("Total training {} images".format(str(img_num)))
        if count > 1000:
            break

    count = 0
    for images, labels in imgnet_valid.Dataset:
        valid_step(images, labels)
        count += 1
        if count > 100:
            break
    
    template = "Epoch {}, Loss: {:4.3f}, Accuracy: {:4.3f}, Valid Loss: {:4.3f}, Valid Accuracy: {:4.3f}"
    writetemplate = "Epoch {}, Loss: {:4.3f}, Accuracy: {:4.3f}, Valid Loss: {:4.3f}, Valid Accuracy: {:4.3f}\n"
    print("")
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result(),
                          valid_loss.result(), valid_accuracy.result()))
    log.write(writetemplate.format(epoch+1, train_loss.result(), train_accuracy.result(),
                          valid_loss.result(), valid_accuracy.result()))
    
    if valid_loss.result().numpy() < best_loss:
        best_loss = valid_loss.result().numpy()
        model.save_weights(weights_path)
        print("Model saved! with loss: {:4.3f}".format(valid_loss.result().numpy()))
        log.write("Model saved! with loss: {:4.3f}\n".format(valid_loss.result().numpy()))
        print("")
    log.write("   \n")
    
print("Training End!")
log.write("Training End!\n")

yolo_model = tf.keras.Model([model.get_layer("Input_stage").input], 
                            [model.get_layer("tf_op_layer_add_10").output,
                             model.get_layer("tf_op_layer_add_18").output,
                             model.get_layer("tf_op_layer_add_22").output])

yolo_model.save_weights("weights/yolo_darknet_weights.h5")

log.write("Transfer darknet weights success!\n")
log.close()