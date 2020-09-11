import os
import sys
import time
import random
import string
import argparse
import numpy as np
import tensorflow as tf
from models import yolov3
from gpus import gpu_growth

parser = argparse.ArgumentParser()


parser.add_argument("--GPU", type=str, default="0", help="Specify the GPU number")

parser.add_argument("--load_path", type=str, default="weights/yolov3.weights", help="load darknet weights path")
parser.add_argument("--save_path", type=str, default="weights/yolov3.h5", help="Output tf weights path")

opt = parser.parse_args()

gpu_growth(opt.GPU)


yolov3_model = yolov3.Darknet_yolo()
yolov3.load_weights(opt.load_path, yolov3_model)

yolov3_model.save(opt.save_path)

print("Transform weights complete!")