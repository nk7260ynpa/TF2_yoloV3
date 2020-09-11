import os
import tensorflow as tf

def gpu_growth(gpu_num):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
	gpus = tf.config.experimental.list_physical_devices("GPU")
	for i in range(len(gpus)):
		tf.config.experimental.set_memory_growth(gpus[i], True)
	print("Total {} GPUS".format(len(gpus)))
	return len(gpus)