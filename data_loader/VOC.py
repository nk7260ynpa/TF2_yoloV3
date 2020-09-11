import os
import pathlib
import numpy as np
import tensorflow as tf
import datetime
import glob

def path_to_tensor(file_path):
    strings = tf.io.read_file(file_path)
    return tf.io.parse_tensor(strings, out_type=tf.float32)

def read_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (416, 416), method="nearest")
    img /= 255.
    return img

def x_y_preprocess(data_path, label_l, label_m, label_s):
    img = read_image(data_path)
    tensor_l = path_to_tensor(label_l)
    tensor_m = path_to_tensor(label_m)
    tensor_s = path_to_tensor(label_s)
    return img, tensor_l, tensor_m, tensor_s

def VOC_dataset(DATASET_PATH, LABEL_PATH):
	data_root = pathlib.Path(os.path.join(DATASET_PATH, "JPEGImages"))
	data_path_list = [str(path) for path in data_root.glob("./*.jpg")]
	label_l_list = [os.path.join(LABEL_PATH, str(path.name)[:-4]+"_l.ts") for path in data_root.glob("./*.jpg")]
	label_m_list = [os.path.join(LABEL_PATH, str(path.name)[:-4]+"_m.ts") for path in data_root.glob("./*.jpg")]
	label_s_list = [os.path.join(LABEL_PATH, str(path.name)[:-4]+"_s.ts") for path in data_root.glob("./*.jpg")]
	ds = tf.data.Dataset.from_tensor_slices((data_path_list, label_l_list, label_m_list, label_s_list))
	ds = ds.map(x_y_preprocess)
	ds = ds.shuffle(buffer_size=50).repeat().batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	return ds
	