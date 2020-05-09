import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from imagenet_info import imagenet_info
from imagenet import imagenet_info, imagenet_train, imagenet_valid
from Darknet53_model import Darknet53
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "12,13"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)


strategy = tf.distribute.MirroredStrategy()

imgnet_info  = imagenet_info("map_clsloc.txt", "ILSVRC2012_validation_ground_truth.txt")
imgnet_train = imagenet_train("/raid/peterchen/datasets/imagenet/2012/train/", 256, imgnet_info, batch_size=256)
imgnet_valid = imagenet_valid("/raid/peterchen/datasets/imagenet/2012/valid/", 256, imgnet_info, batch_size=256)
imgnet_train_dist_dataset = strategy.experimental_distribute_dataset(imgnet_train.Dataset)
imgnet_valid_dist_dataset = strategy.experimental_distribute_dataset(imgnet_valid.Dataset)

with strategy.scope():
    model = Darknet53()
    
with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=256)
    optimizer = tf.keras.optimizers.Adam(0.0001)
    
with strategy.scope():
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    valid_loss = tf.keras.metrics.Mean(name="valid_loss")
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")
    
#@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss.update_state(loss)
    train_accuracy(labels, predictions)

#@tf.function
def valid_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)
    valid_loss.update_state(loss)
    valid_accuracy.update_state(labels, predictions)

@tf.function
def distributed_train_step(images, labels):
    strategy.experimental_run_v2(train_step, (images, labels))

    
@tf.function
def distributed_valid_step(dataset_inputs):
    strategy.experimental_run_v2(valid_step, (images, labels))

time = datetime.datetime.now()
#weight_name = "weights"+'_'+str(time.month)+'-'+str(time.day)+'/'
weight_name = "mul_weights.h5"
weights_path = os.path.join("weights", weight_name)
#if not os.path.exists(weights_path):
#    os.mkdir(weights_path)

EPOCHS = 50
img_num = 0
best_loss = np.inf
log = open(f'./train_mul_record.txt', 'a')
print("Training Start!")
log.write("Training Start!\n")
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    
    count = 0
    for images, labels in imgnet_train_dist_dataset:
        distributed_train_step(images, labels)
        count += 1
        img_num += 256
        if count % 500 == 0:
            print("Total training {} images".format(str(img_num)))
        if count > 1000:
            break

    count = 0
    for images, labels in imgnet_valid_dist_dataset:
        distributed_valid_step(images, labels)
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
log.close()
    
