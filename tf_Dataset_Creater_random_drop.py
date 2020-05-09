import pathlib
import os
import tensorflow as tf
import numpy as np
import random
import tensorflow_addons as tfa

class tf_data_gn():
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_root = pathlib.Path(self.data_path)
        self.label_names, self.label_to_index = self.label_names_gn()
        
    # convert image to nparray 
    def preprocessing_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Data augmentation
        image = tf.image.resize(image, (256, 256))
        crop_size = tf.random.uniform([1], minval=192, maxval=256, dtype=tf.int32)
        image = tf.image.random_crop(image , [crop_size[0], crop_size[0], 3])
        image = tf.image.resize(image, (224, 224))
        
        # blur
        random_matrix = tf.random.uniform((224, 224, 1), minval=0, maxval=1)
        random_matrix = tf.cast(random_matrix > 0.1, tf.float32)
        image = image * random_matrix
     
        # Normalize
        image /= 255,
        return image
    
    def valid_preprocessing_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Data augmentation
        image = tf.image.resize(image, (224, 224))

        # Normalize
        image /= 255,
        return image
    
    # read image from each path
    def load_and_preprocessing_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocessing_image(image)
    
    def valid_load_and_preprocessing_image(self, path):
        image = tf.io.read_file(path)
        return self.valid_preprocessing_image(image)
        
    # generate label name and label index dictionary    
    def label_names_gn(self):
        label_names = sorted(item.name for item in self.data_root.glob("*/") if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        return label_names, label_to_index
    
    # generate_dataset
    
    def dataset_generate(self, all_image_paths, batch_size):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        all_image_labels = [self.label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(self.load_and_preprocessing_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.float32))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        image_counts = len(all_image_paths)
        image_label_ds = image_label_ds.shuffle(buffer_size=6000).repeat().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        return image_label_ds
    
    def valid_dataset_generate(self, all_image_paths, batch_size):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        all_image_labels = [self.label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(self.valid_load_and_preprocessing_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.float32))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        image_counts = len(all_image_paths)
        image_label_ds = image_label_ds.shuffle(buffer_size=6000).repeat().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        return image_label_ds
    
    # generate_train_and_test_dataset
    def dataset(self, batch_size, shuffle=True, valid_split=True, valid_split_rate=0.1):
        
        all_image_paths = list(self.data_root.glob("*/*"))
        all_image_paths = [str(path) for path in all_image_paths]

        if shuffle == True:
            random.shuffle(all_image_paths)
        
        if valid_split == True:
            valid_len = int(np.around(valid_split_rate*len(all_image_paths)))
            train_image_paths = all_image_paths[valid_len:]
            self.train_step = int(len(train_image_paths) / batch_size)
            print(self.train_step)
            valid_image_paths = all_image_paths[:valid_len]
            self.valid_step = int(len(valid_image_paths) / batch_size)
            print(self.valid_step)
            return self.dataset_generate(train_image_paths, batch_size), self.valid_dataset_generate(valid_image_paths, batch_size)
        else:
            return self.dataset_generate(all_image_paths, batch_size)
        

    
        