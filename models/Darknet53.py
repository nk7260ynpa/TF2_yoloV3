import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

def DarkConv(inputs, filters, kernel_size, layer_idx, name, strides=(1, 1), padding="SAME"):
    layer_name = "{}_layer_{}".format(name, str(layer_idx))
    inputs = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, 
                    name=layer_name+"_Conv")(inputs)
    inputs = BatchNormalization(momentum=0.9, epsilon=1e-05, name=layer_name+"_BN")(inputs)
    inputs = LeakyReLU(alpha=0.1, name=layer_name+"_Activation")(inputs)
    return inputs

def DarkPool(inputs, filters, layer_idx, name, strides=(2, 2), padding="VALID"):
    layer_name = "{}_layer_{}".format(name, str(layer_idx))
    inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)), name=layer_name)(inputs)
    inputs = DarkConv(inputs, filters, (3, 3), layer_idx, name, strides, padding)
    return inputs

def ResidualBlock(inputs, filters, layer_idx, name, strides=(1, 1), padding="SAME"):
    layer1, layer2 = layer_idx
    shortcut = inputs
    inputs = DarkConv(inputs, filters   , (1, 1), layer1, name)
    inputs = DarkConv(inputs, filters*2 , (3, 3), layer2, name)
    inputs = shortcut + inputs
    return inputs

def Darknet53(input_size, classes):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3,), name="Input_stage")

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
    x = DarkPool(x , 512, 27, "stage3")

    x = ResidualBlock(x, 256, [28, 29], "stage4")
    x = ResidualBlock(x, 256, [30, 31], "stage4")
    x = ResidualBlock(x, 256, [32, 33], "stage4")
    x = ResidualBlock(x, 256, [34, 35], "stage4")
    x = ResidualBlock(x, 256, [36, 37], "stage4")
    x = ResidualBlock(x, 256, [38, 39], "stage4")
    x = ResidualBlock(x, 256, [40, 41], "stage4")
    x = ResidualBlock(x, 256, [42, 43], "stage4")
    x = DarkPool(x , 1024, 44, "stage4")

    x = ResidualBlock(x, 512, [45, 46], "stage5")
    x = ResidualBlock(x, 512, [47, 48], "stage5")
    x = ResidualBlock(x, 512, [49, 50], "stage5")
    x = ResidualBlock(x, 512, [51, 52], "stage5")

    x = tf.keras.layers.GlobalAveragePooling2D(name="stage6_Average_Pooling")(x)

    x = tf.keras.layers.Dense(classes, activation=None, name="stage6_layer_53_Dense", use_bias=True)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs, x, name="Darknet")
    return model


# def Darknet53_yolo():
#     inputs = tf.keras.Input(shape=(416, 416, 3,), name="Input_stage_yolo")

#     x = DarkConv(inputs, 32, (3, 3), 1, "stage0")
#     x = DarkPool(x , 64,  2, "stage0")

#     x = ResidualBlock(x, 32, [3, 4], "stage1")
#     x = DarkPool(x , 128, 5, "stage1")

#     x = ResidualBlock(x, 64, [6, 7], "stage2")
#     x = ResidualBlock(x, 64, [8, 9], "stage2")
#     x = DarkPool(x , 256, 10, "stage2")

#     x = ResidualBlock(x, 128, [11, 12], "stage3")
#     x = ResidualBlock(x, 128, [13, 14], "stage3")
#     x = ResidualBlock(x, 128, [15, 16], "stage3")
#     x = ResidualBlock(x, 128, [17, 18], "stage3")
#     x = ResidualBlock(x, 128, [19, 20], "stage3")
#     x = ResidualBlock(x, 128, [21, 22], "stage3")
#     x = ResidualBlock(x, 128, [23, 24], "stage3")
#     x = ResidualBlock(x, 128, [25, 26], "stage3")
#     output_1 = x
#     x = DarkPool(x , 512, 27, "stage3")

#     x = ResidualBlock(x, 256, [28, 29], "stage4")
#     x = ResidualBlock(x, 256, [30, 31], "stage4")
#     x = ResidualBlock(x, 256, [32, 33], "stage4")
#     x = ResidualBlock(x, 256, [34, 35], "stage4")
#     x = ResidualBlock(x, 256, [36, 37], "stage4")
#     x = ResidualBlock(x, 256, [38, 39], "stage4")
#     x = ResidualBlock(x, 256, [40, 41], "stage4")
#     x = ResidualBlock(x, 256, [42, 43], "stage4")
#     output_2 = x
#     x = DarkPool(x , 1024, 44, "stage4")

#     x = ResidualBlock(x, 512, [45, 46], "stage5")
#     x = ResidualBlock(x, 512, [47, 48], "stage5")
#     x = ResidualBlock(x, 512, [49, 50], "stage5")
#     x = ResidualBlock(x, 512, [51, 52], "stage5")
#     output_3 = x

#     model = tf.keras.Model(inputs, [output_1, output_2, output_3], name="Darknet_yolo")
#     return model