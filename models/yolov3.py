import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from .Darknet53 import  DarkConv, DarkPool, ResidualBlock
import numpy as np

def Darknet53_yolo_base():
    inputs = tf.keras.Input(shape=(416, 416, 3,), name="Input_stage_yolo")

    x = DarkConv(inputs, 32, (3, 3), 1, "stage0")
    x = DarkPool(x , 64,  2, "stage0")

    x = ResidualBlock(x, 32, [3, 4], "stage1")
    x = DarkPool(x , 128, 5, "stage1")

    x = ResidualBlock(x, 64, [6, 7], "stage2")
    x = ResidualBlock(x, 64, [8, 9], "stage2")
    x = DarkPool(x , 256, 10, "stage2")

    x = ResidualBlock(x, 128, [11, 12], "stage3")
    x = ResidualBlock(x, 128, [13, 14], "stage3")
    x = ResidualBlock(x, 128, [15, 16], "stage3")
    x = ResidualBlock(x, 128, [17, 18], "stage3")
    x = ResidualBlock(x, 128, [19, 20], "stage3")
    x = ResidualBlock(x, 128, [21, 22], "stage3")
    x = ResidualBlock(x, 128, [23, 24], "stage3")
    x = ResidualBlock(x, 128, [25, 26], "stage3")
    output_1 = x
    x = DarkPool(x , 512, 27, "stage3")

    x = ResidualBlock(x, 256, [28, 29], "stage4")
    x = ResidualBlock(x, 256, [30, 31], "stage4")
    x = ResidualBlock(x, 256, [32, 33], "stage4")
    x = ResidualBlock(x, 256, [34, 35], "stage4")
    x = ResidualBlock(x, 256, [36, 37], "stage4")
    x = ResidualBlock(x, 256, [38, 39], "stage4")
    x = ResidualBlock(x, 256, [40, 41], "stage4")
    x = ResidualBlock(x, 256, [42, 43], "stage4")
    output_2 = x
    x = DarkPool(x , 1024, 44, "stage4")

    x = ResidualBlock(x, 512, [45, 46], "stage5")
    x = ResidualBlock(x, 512, [47, 48], "stage5")
    x = ResidualBlock(x, 512, [49, 50], "stage5")
    x = ResidualBlock(x, 512, [51, 52], "stage5")
    output_3 = x

    model = tf.keras.Model(inputs, [output_1, output_2, output_3], name="Darknet_yolo")
    return model

def Efficient_yolo_base():
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(416, 416, 3), include_top=False, weights="imagenet")
    inputs = base_model.input
    outputs = [base_model.get_layer("block4a_expand_activation").output,
               base_model.get_layer("block6a_expand_activation").output,
               base_model.get_layer("top_activation").output]
    model = tf.keras.Model(inputs, outputs)
    return model


def yolo_block(inputs, filters, name):
    layer_idx = iter(range(1, 7))
    inputs = DarkConv(inputs, filters, kernel_size=1, layer_idx=next(layer_idx), name=name)
    inputs = DarkConv(inputs, filters*2, kernel_size=3, layer_idx=next(layer_idx), name=name)
    inputs = DarkConv(inputs, filters, kernel_size=1, layer_idx=next(layer_idx), name=name)
    inputs = DarkConv(inputs, filters*2, kernel_size=3, layer_idx=next(layer_idx), name=name)
    inputs = DarkConv(inputs, filters, kernel_size=1, layer_idx=next(layer_idx), name=name)
    route = inputs 
    inputs = DarkConv(inputs, filters*2, kernel_size=3, layer_idx=next(layer_idx), name=name)
    return route, inputs 

def upsample(inputs, output_shape):
    inputs = tf.image.resize(inputs, output_shape, method="nearest")
    inputs = tf.identity(inputs, "upsampled")
    return inputs

def Darknet_yolo(num_class=80):
    start_model = tf.keras.layers.Input(shape=(416, 416, 3))
    conv_yolo = Darknet53_yolo_base()
    conv_output_1, conv_output_2, conv_output_3 = conv_yolo(start_model)
    
    #route1
    start = tf.keras.layers.Input(shape=(13, 13, 1024), name="yolo_block1_Input")
    route, inputs = yolo_block(start, 512, name="yolo_block_1")
    detect = Conv2D(filters=3 * (5+num_class), kernel_size=1, strides=1, name="yolo_block_1_output_Conv")(inputs)
    yolo_block_1_model = tf.keras.Model(inputs=start, outputs=[route, detect], name="yolo_block_1")
    
    #route2
    start1 = tf.keras.layers.Input(shape=(13, 13, 512), name="yolo_block2_Input")
    route = DarkConv(start1, filters=256, kernel_size=1, strides=(1, 1), padding="SAME", layer_idx=0, name="yolo_block_2")
    route = upsample(route, (26, 26))
    start2 = tf.keras.layers.Input(shape=(26, 26, 512))
    inputs = tf.concat([route, start2], axis=3)
    route, inputs = yolo_block(inputs, 256, name="yolo_block_2")
    detect = Conv2D(filters=3 * (5+num_class), kernel_size=1, strides=1, name="yolo_block_2_output_Conv")(inputs)
    yolo_block_2_model = tf.keras.Model(inputs=[start1, start2], outputs=[route, detect], name="yolo_block_2")
    
    #route3
    start1 = tf.keras.layers.Input(shape=(26, 26, 256), name="yolo_block_3_Input")
    route = DarkConv(start1, filters=128, kernel_size=1, strides=(1, 1), padding="SAME", layer_idx=0, name="yolo_block_3")
    route = upsample(route, (52, 52))
    start2 = tf.keras.layers.Input(shape=(52, 52, 256))
    inputs = tf.concat([route, start2], axis=3)
    route, inputs = yolo_block(inputs, 128, name="yolo_block_3")
    detect = Conv2D(filters=3 * (5+num_class), kernel_size=1, strides=1, name="yolo_block_3_output_Conv")(inputs)
    yolo_block_3_model = tf.keras.Model(inputs=[start1, start2], outputs=[detect], name="yolo_block_3")
    
    #output
    route, detect_1 = yolo_block_1_model(conv_output_3)
    route, detect_2 = yolo_block_2_model([route, conv_output_2])
    detect_3 = yolo_block_3_model([route, conv_output_1])
    
    yolo_model = tf.keras.Model(start_model, [detect_1, detect_2, detect_3])
    return yolo_model

## ====================####

def Efficient_yolo(num_class=80):
    start_model = tf.keras.layers.Input(shape=(416, 416, 3))
    conv_yolo = Efficient_yolo_base()
    conv_output_1, conv_output_2, conv_output_3 = conv_yolo(start_model)
    
    #route1
    start = tf.keras.layers.Input(shape=(13, 13, 1280), name="yolo_block1_Input")
    route, inputs = yolo_block(start, 640, name="yolo_block_1")
    detect = Conv2D(filters=3 * (5+num_class), kernel_size=1, strides=1, name="yolo_block_1_output_Conv")(inputs)
    yolo_block_1_model = tf.keras.Model(inputs=start, outputs=[route, detect], name="yolo_block_1")
    
    #route2
    start1 = tf.keras.layers.Input(shape=(13, 13, 640), name="yolo_block2_Input")
    route = DarkConv(start1, filters=320, kernel_size=1, strides=(1, 1), padding="SAME", layer_idx=0, name="yolo_block_2")
    route = upsample(route, (26, 26))
    start2 = tf.keras.layers.Input(shape=(26, 26, 672))
    inputs = tf.concat([route, start2], axis=3)
    route, inputs = yolo_block(inputs, 320, name="yolo_block_2")
    detect = Conv2D(filters=3 * (5+num_class), kernel_size=1, strides=1, name="yolo_block_2_output_Conv")(inputs)
    yolo_block_2_model = tf.keras.Model(inputs=[start1, start2], outputs=[route, detect], name="yolo_block_2")
    
    #route3
    start1 = tf.keras.layers.Input(shape=(26, 26, 320), name="yolo_block_3_Input")
    route = DarkConv(start1, filters=160, kernel_size=1, strides=(1, 1), padding="SAME", layer_idx=0, name="yolo_block_3")
    route = upsample(route, (52, 52))
    start2 = tf.keras.layers.Input(shape=(52, 52, 240))
    inputs = tf.concat([route, start2], axis=3)
    route, inputs = yolo_block(inputs, 128, name="yolo_block_3")
    detect = Conv2D(filters=3 * (5+num_class), kernel_size=1, strides=1, name="yolo_block_3_output_Conv")(inputs)
    yolo_block_3_model = tf.keras.Model(inputs=[start1, start2], outputs=[detect], name="yolo_block_3")
    
    #output
    route, detect_1 = yolo_block_1_model(conv_output_3)
    route, detect_2 = yolo_block_2_model([route, conv_output_2])
    detect_3 = yolo_block_3_model([route, conv_output_1])
    
    yolo_model = tf.keras.Model(start_model, [detect_1, detect_2, detect_3])
    return yolo_model




def detection_layer(model_output, img_size, anchors, num_classes):
    num_anchors = len(anchors)
    shape = model_output.shape.as_list()
    grid_size = shape[1:3]
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes
    predictions = tf.reshape(model_output, [-1, num_anchors * dim, bbox_attrs])
    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    #anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
    anchors = [(float(a[0]), float(a[1])) for a in anchors]
    #print(anchors)
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
    
    box_centers = tf.nn.sigmoid(box_centers)
    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride
    
    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    #box_sizes = box_sizes * stride
    
    confidence = tf.nn.sigmoid(confidence)
    
    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    
    classes = tf.nn.sigmoid(classes)
    
    predictions = tf.concat([detections, classes], axis=-1)
    return predictions


def coco_pretrained_weights(weights_file, model):
    global ptr
    
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)
    i = 0
    while i < len(model.variables) - 1:
        var1 = model.variables[i]
        var2 = model.variables[i+1]
        if "Conv" in var1.name:
            #print(var1.name.split('/')[-2])
            if "BN" in var2.name:
                gamma, beta, mean, var = model.variables[i+1: i+5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    var.assign(var_weights)
                i += 4   
                
            elif "Conv" in var2.name:
                #print(var2.name.split('/'))
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr+bias_params].reshape(bias_shape)
                ptr += bias_params
                bias.assign(bias_weights)
                i += 1    
            
            
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr+num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            var1.assign(var_weights)
            i += 1


            
def load_weights(weights_file, yolov3_model):
    global ptr
    ptr = 0
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("Darknet_yolo"))
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("yolo_block_1"))
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("yolo_block_2"))
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("yolo_block_3"))
            




