import os 
import numpy as np
import tensorflow as tf

def Random_drop(img, drop_rate):
    if drop_rate < 0. or drop_rate > 1.:
        assert False
    drop = tf.random.uniform((img.shape[0], img.shape[1]), minval=0, maxval=1)
    drop = tf.expand_dims(tf.cast(drop > drop_rate, tf.float32), axis=2)
    return img * drop

def Resize_drop(img, img_size):
    img_w = tf.cast((tf.shape(img)[1]), tf.float32)
    img_h = tf.cast((tf.shape(img)[0]), tf.float32)
    short_side = tf.where(img_h>img_w, x=img_w, y=img_h)
    resize_rate = tf.cast(img_size, tf.float32) / short_side
    if img_h > img_w:
        img = tf.image.resize(img, (tf.math.ceil(img_h*resize_rate), img_size))
    else:
        img = tf.image.resize(img, (img_size, tf.math.ceil(img_w*resize_rate)))
    img = tf.image.random_crop(img, [img_size, img_size, 3])
    return img

def Square_drop(img, drop_size=0.2):
    if drop_size < 0. or drop_size > 1.:
        assert False
    
    img_w = img.shape[1]
    img_h = img.shape[0]
    
    block_w = int(img_w * drop_size)
    block_h = int(img_h * drop_size)
    block = tf.zeros([block_h, block_w], dtype=tf.float32)
    pad_w = img_w - block_w
    pad_h = img_h - block_h
    random_w = tf.random.uniform([1], minval=0, maxval=pad_w, dtype=tf.int32)
    random_h = tf.random.uniform([1], minval=0, maxval=pad_h, dtype=tf.int32)
    
    filter_block = tf.pad(block, [(pad_h-random_h[0], random_h[0]), (pad_w-random_w[0], random_w[0])], constant_values=1.)
    filter_block = tf.expand_dims(filter_block, axis=2)
    img = img * filter_block
    return img
    
def Scale_to_one(img):
    return tf.cast(img, tf.float32) / 255.

def Standardize(img):
    img -= 0.5
    img /= 0.5
    return img

def Resize(img, img_size):
    return tf.image.resize(img, (img_size, img_size))


    