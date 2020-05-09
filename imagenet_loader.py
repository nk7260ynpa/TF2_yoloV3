import os
import tensorflow as tf
import pathlib


class train_image_dataset():
    #super(train_image_dataset, self).__init__()
    def __init__(self, img_folder, img_size, imagenet_info, batch_size=32):
        self.img_folder = img_folder
        self.img_size = img_size
        self.batch_size = batch_size
        self.index_to_net_label = imagenet_info.index_to_net_label
        #self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.Dataset = self._generator()
    
    def train_preprocessing_image(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize(img, (self.img_size+32, self.img_size+32))
        crop_size = tf.random.uniform([1], minval=self.img_size-32, maxval=self.img_size+32, dtype=tf.int32)
        img = tf.image.random_crop(img , [crop_size[0], crop_size[0], 3])
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img /= 255.
        return img
    
    def _generator(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_root = pathlib.Path(self.img_folder)
        all_image_path = []
        all_label = []
        for path in data_root.glob("./*/*"):
            all_image_path.append(str(path))
            all_label.append(self.index_to_net_label[path.parent.name])
        
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
        image_ds = path_ds.map(self.train_preprocessing_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(all_label)
       
        path_image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        path_image_label_ds = path_image_label_ds.shuffle(buffer_size=6000).repeat().\
        batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        self.data_length = len(all_image_path)
        return path_image_label_ds
    
    def __len__(self):
        return self.data_length
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(iter(self.Dataset))
    


class valid_image_dataset():
    def __init__(self, img_folder, img_size, imagenet_info, batch_size=100):
        self.img_folder = img_folder
        self.img_size = img_size
        self.batch_size = batch_size
        self.index_to_net_label = imagenet_info.index_to_net_label
        self.imgnet_info = imagenet_info
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.Dataset = self._generator()
    
    def valid_preprocessing_image(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img /= 255.
        return img 
    
    def _generator(self):
        all_images = os.listdir(self.img_folder)
        all_images.sort()
        all_image_path = [os.path.join(self.img_folder, file) for file in all_images]
        all_label = self.imgnet_info.val_label["net_label"]
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
        image_ds = path_ds.map(self.valid_preprocessing_image, num_parallel_calls=self.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(all_label)
        path_image_label_ds = tf.data.Dataset.zip((path_ds, image_ds, label_ds))
        path_image_label_ds = path_image_label_ds.shuffle(buffer_size=12000).repeat().\
        batch(self.batch_size).prefetch(buffer_size=self.AUTOTUNE)
        self.data_length = len(all_image_path)
        return path_image_label_ds
    
    def __len__(self):
        return self.data_length
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(iter(self.Dataset))