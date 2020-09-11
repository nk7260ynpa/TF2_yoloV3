import numpy as np
import pandas as pd
import os
import tensorflow as tf
import pathlib
import random

class imagenet_info():
    def __init__(self):
        self.map_index_path = "data/imagenet/map_clsloc.txt"
        self.val_label_path = "data/imagenet/ILSVRC2012_validation_ground_truth.txt"
        self.df = self._df_generator()
        self.val_label = self._valid_label_generator()
        self.index_to_net_label = dict(zip(self.df["index"], self.df["net_label"]))
        self.label_to_net_label = dict(zip(self.val_label["label"], self.val_label["net_label"]))
        self.net_label_to_index = dict(zip(self.val_label["net_label"], self.val_label["index"]))
    
    def _df_generator(self):
        with open(self.map_index_path) as f:
            map_index = f.readlines()
        df = pd.DataFrame(map_index, columns=["all_label"])
        df["index"] = df.all_label.str.split(" ").str.get(0)
        df["label"] = df.all_label.str.split(" ").str.get(1).astype(np.int32)
        df["name"] = df.all_label.str.split(" ").str.get(2).str.strip("\n")
        df["net_label"] = df["label"] - 1
        df = df.drop("all_label", axis=1)
        df = df[["net_label", "label", "index", "name"]]
        return df
    
    def _valid_label_generator(self):
        with open(self.val_label_path) as f:
            val_label = f.readlines()
        val_label_df = pd.DataFrame(val_label, columns=["label"])
        val_label_df["label"] = val_label_df["label"].str.strip("\n").astype(np.int32)
        image_num = [i for i in range(1, 50001)]
        val_label_df["image_num"] = image_num
        val_label_df = val_label_df.merge(self.df, left_on='label', right_on='label')
        val_label_df = val_label_df.sort_values("image_num")
        val_label_df = val_label_df.reset_index(drop=True)
        val_label_df = val_label_df[["image_num", "net_label", "label", "index", "name"]]
        return val_label_df 
      
class imagenet_train():
    #super(train_image_dataset, self).__init__()
    def __init__(self, img_folder, img_size, preprocess, batch_size=32, buffer_size=12000):
        self.img_folder = img_folder
        self.img_size = img_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.imgnet_info = imagenet_info()
        self.classes = 1000
        self.preprocess = preprocess
        self.index_to_net_label = self.imgnet_info.index_to_net_label
        self.Dataset = self._generator()
    
    def train_preprocessing_image(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = self.preprocess(img)
        return img
    
    def _generator(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_root = pathlib.Path(self.img_folder)
        all_image_path = []
        all_label = []

        all_image_libpath = [path for path in data_root.glob("./*/*.JPEG")]
        random.shuffle(all_image_libpath)
        for path in all_image_libpath:
            all_image_path.append(str(path))
            all_label.append(self.index_to_net_label[path.parent.name])
            
        self.data_length = len(all_image_path)
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
        image_ds = path_ds.map(self.train_preprocessing_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(all_label)
        
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        image_label_ds = image_label_ds.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True).repeat().\
        batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        self.data_length = len(all_image_path)
        return image_label_ds
    
    def __len__(self):
        return self.data_length
    

class imagenet_valid():
    def __init__(self, img_folder, img_size, preprocess, batch_size=32):
        self.img_folder = img_folder
        self.img_size = img_size
        self.batch_size = batch_size
        self.imgnet_info = imagenet_info()
        self.preprocess = preprocess
        self.index_to_net_label = self.imgnet_info.index_to_net_label
        self.Dataset = self._generator()
    
    def valid_preprocessing_image(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = self.preprocess(img)
        return img
    
    def _generator(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        all_images = os.listdir(self.img_folder)
        all_images.sort()
        all_image_path = [os.path.join(self.img_folder, file) for file in all_images]
        all_label = self.imgnet_info.val_label["net_label"]
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
        image_ds = path_ds.map(self.valid_preprocessing_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(all_label)
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        image_label_ds = image_label_ds.repeat().\
        batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        self.data_length = len(all_image_path)
        return image_label_ds
    
    def __len__(self):
        return self.data_length
        
