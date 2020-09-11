import os
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from gpus import gpu_growth
import time
import random
import string
import argparse
from data_loader.imagenet import imagenet_train, imagenet_valid
from models.Darknet53 import Darknet53
from process import Scale_to_one, Standardize, Random_drop, Square_drop, Resize, Resize_drop


def train_process(img, img_size=224,drop_rate=0.0, Square_rate=0.0):
    img = Scale_to_one(img)
    img = Standardize(img)
    img = Resize_drop(img, img_size)
    if drop_rate != 0.0 :
        img = Random_drop(img, drop_rate)
    
    if Square_rate != 0.0:
        img = Square_drop(img, Square_rate)
    return img

def valid_process(img, img_size=224):
    img = Scale_to_one(img)
    img = Standardize(img)
    img = Resize(img, img_size)
    return img

def model_maker(model_name, input_size, classes, pretrained_path):
    if model_name == "Darknet53":
        model = Darknet53(input_size, classes)
        if pretrained_path != None:
            model.load_weights(pretrained_path)
        return model
    else:
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(input_size, input_size,3), 
                                                          weights='imagenet', include_top=False)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(classes, activation=None)(x)
        predictions = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions) 
        if pretrained_path != None:
            model.load_weights(pretrained_path)
        return model
        
@tf.function
def train_step(images, labels, loss_object, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def valid_step(images, labels, loss_object, valid_loss, valid_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)
    valid_loss(loss)
    valid_accuracy(labels, predictions)
    
    
# mul training

def mul_train_step(images, labels, compute_loss, train_loss, train_accuracy, batch_size):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions, batch_size)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)


def mul_valid_step(images, labels, compute_loss, valid_loss, valid_accuracy, batch_size):
    with tf.GradientTape() as tape:
        predictions = model(images, training=False)
        loss = compute_loss(labels, predictions, batch_size)
    valid_loss(loss)
    valid_accuracy(labels, predictions)

@tf.function
def distributed_train_step(images, labels, compute_loss, train_loss, train_accuracy, batch_size):
    strategy.run(mul_train_step, (images, labels, compute_loss, train_loss, train_accuracy, batch_size))
    
@tf.function
def distributed_valid_step(images, labels, compute_loss, train_loss, train_accuracy, batch_size):
    strategy.run(mul_valid_step, (images, labels, compute_loss, train_loss, train_accuracy, batch_size))

parser = argparse.ArgumentParser()
# System
parser.add_argument("--GPU", type=str, default="0", help="Specify the GPU number")

# Dataset
parser.add_argument("--dataset", type=str, default="imagenet", help="Imagenet or Custom dataset")
parser.add_argument("--data_split", type=bool, default=False, help="Split data to train and valid")
parser.add_argument("--split_rate", type=float, default=0.15, help="Data split rate")
parser.add_argument("--train_path", type=str, default="data/imagenet/train/", help="Train data path")
parser.add_argument("--valid_path", type=str, default="data/imagenet/valid/", help="Valid data path")

# Model
parser.add_argument("--model", type=str, default="Darknet53", choices=["Darknet53", "Efficientnet"], 
                    help="Darknet53 or Efficientnet")
parser.add_argument("--pretrained", type=str, help="Pretrained weight path")
parser.add_argument("--weight_name", type=str, help="Weight name")

# Training
parser.add_argument("--batch_size", type=int, default=120, help="Batch_size per every step")
parser.add_argument("--steps", type=int, default=1000, help="Training steps per every epoch")
parser.add_argument("--valid_steps", type=int, default=1000, help="Valid steps per every epoch")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--patient", type=int, default=30, help="Callback patient")
parser.add_argument("--input_size", type=int, default=224, help="Input image size")
parser.add_argument("--buffer_size", type=int ,default=15000, help="Buffer size")

opt = parser.parse_args()

gpu_num = gpu_growth(opt.GPU)

if gpu_num == 1:

    # Dataset
    imgnet_train = imagenet_train(opt.train_path, opt.batch_size, train_process, 
                                  batch_size=opt.batch_size, buffer_size=opt.buffer_size)
    imgnet_valid = imagenet_valid(opt.valid_path, opt.batch_size, valid_process, batch_size=opt.batch_size)
    train_dataset = imgnet_train.Dataset
    valid_dataset = imgnet_valid.Dataset

    # Model
    model = model_maker(opt.model, opt.input_size, imgnet_train.classes, opt.pretrained)

    # Training setting
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(opt.lr)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    valid_loss = tf.keras.metrics.Mean(name="valid_loss")
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")

    # weight name
    if opt.weight_name != None:
        weight_name = opt.weight_name
    else:
        weight_name = opt.model + ".h5"
    weights_path = os.path.join("weights", weight_name)


    EPOCHS = opt.epochs

    best_loss = np.inf
    img_num = 0

    if opt.model == "Darknet53":
        log = open(f'./log/Darknet53_train_record.txt', 'a')
    else:
        log = open(f'./log/Efficient_train_record.txt', 'a')


    print("Training Start!")
    log.write("Training Start!\n")
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        count = 0
        for images, labels in imgnet_train.Dataset:
            train_step(images, labels, loss_object, train_loss, train_accuracy)
            count += 1
            img_num += opt.batch_size
            if count > opt.steps:
                break

        count = 0
        for images, labels in imgnet_valid.Dataset:
            valid_step(images, labels, loss_object, valid_loss, valid_accuracy)
            count += 1
            if count > opt.valid_steps:
                break
        print("")
        print("Total training {} images".format(str(img_num)))
        log.write("Total training {} images\n".format(str(img_num)))
        template = "Epoch {}, Loss: {:4.3f}, Accuracy: {:4.3f}, Valid Loss: {:4.3f}, Valid Accuracy: {:4.3f}"
        writetemplate = "Epoch {}, Loss: {:4.3f}, Accuracy: {:4.3f}, Valid Loss: {:4.3f}, Valid Accuracy: {:4.3f}\n"
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result(),
                              valid_loss.result(), valid_accuracy.result()))
        log.write(writetemplate.format(epoch+1, train_loss.result(), train_accuracy.result(),
                              valid_loss.result(), valid_accuracy.result()))

        if valid_loss.result().numpy() < best_loss:
            best_epoch = epoch
            best_loss = valid_loss.result().numpy()
            model.save_weights(weights_path)
            print("Model saved! with loss: {:4.3f}".format(valid_loss.result().numpy()))
            log.write("Model saved! with loss: {:4.3f}\n".format(valid_loss.result().numpy()))
            print("")
        log.write("   \n")

        if epoch - opt.patient > best_epoch:
            break

    print("Training End!")
    log.write("Training End!\n")
    log.close()

else:
    strategy = tf.distribute.MirroredStrategy()
    
    # Dataset
    imgnet_train = imagenet_train(opt.train_path, opt.batch_size, train_process, 
                                  batch_size=opt.batch_size, buffer_size=opt.buffer_size)
    imgnet_valid = imagenet_valid(opt.valid_path, opt.batch_size, valid_process, batch_size=opt.batch_size)
    train_dataset = imgnet_train.Dataset
    valid_dataset = imgnet_valid.Dataset
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    with strategy.scope():
        model = model_maker(opt.model, opt.input_size, imgnet_train.classes, opt.pretrained)
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=False,
          reduction=tf.keras.losses.Reduction.NONE)
        def compute_loss(labels, predictions, batch_size):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)
        optimizer = tf.keras.optimizers.Adam(opt.lr)

        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")
    
        # weight name
    if opt.weight_name != None:
        weight_name = opt.weight_name
    else:
        weight_name = opt.model + ".h5"
    weights_path = os.path.join("weights", weight_name)
 

    EPOCHS = opt.epochs

    best_loss = np.inf
    img_num = 0
    
    if opt.model == "Darknet53":
        log = open(f'./log/Darknet53_train_record.txt', 'a')
    else:
        log = open(f'./log/Efficient_train_record.txt', 'a')
    
    print("Training Start!")
    log.write("Training Start!\n")
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        count = 0
        for images, labels in imgnet_train.Dataset:
            distributed_train_step(images, labels, compute_loss, train_loss, train_accuracy, opt.batch_size)
            count += 1
            img_num += opt.batch_size
            if count > opt.steps:
                break

        count = 0
        for images, labels in imgnet_valid.Dataset:
            distributed_valid_step(images, labels, compute_loss, valid_loss, valid_accuracy, opt.batch_size)
            count += 1
            if count > opt.valid_steps:
                break
        print("")
        print("Total training {} images".format(str(img_num)))
        log.write("Total training {} images\n".format(str(img_num)))
        template = "Epoch {}, Loss: {:4.3f}, Accuracy: {:4.3f}, Valid Loss: {:4.3f}, Valid Accuracy: {:4.3f}"
        writetemplate = "Epoch {}, Loss: {:4.3f}, Accuracy: {:4.3f}, Valid Loss: {:4.3f}, Valid Accuracy: {:4.3f}\n"
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result(),
                              valid_loss.result(), valid_accuracy.result()))
        log.write(writetemplate.format(epoch+1, train_loss.result(), train_accuracy.result(),
                              valid_loss.result(), valid_accuracy.result()))

        if valid_loss.result().numpy() < best_loss:
            best_epoch = epoch
            best_loss = valid_loss.result().numpy()
            model.save_weights(weights_path)
            print("Model saved! with loss: {:4.3f}".format(valid_loss.result().numpy()))
            log.write("Model saved! with loss: {:4.3f}\n".format(valid_loss.result().numpy()))
            print("")
        log.write("   \n")

        if epoch - opt.patient > best_epoch:
            break

    print("Training End!")
    log.write("Training End!\n")
    log.close()
