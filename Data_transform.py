import os
import pathlib
import numpy as np
import tensorflow as tf
import glob
from xml.etree.ElementTree import parse
from gpus import gpu_growth
import argparse
import cv2

def PascalVOCDict(class_txt_path):
    with open(os.path.join(class_txt_path)) as f:
        file_list = f.readlines()
    
    class_dict = {}
    for i, label in enumerate(file_list):
        class_dict[label[:-1]] = i
    return class_dict

class PascalVocXmlConverter():
    def __init__(self, fname, class_dict, img_train_size=None):
        self.tree = parse(fname)
        self.img_train_size = img_train_size
        self.class_dict = class_dict
        self.root = self.tree.getroot()
        self.img_name = self.root.find("filename").text
        self.img_height, self.img_width = self.img_size()
        self.bbox = self.Tobbox()
        
    def img_size(self):
        root_size = self.root.find("size")
        height = float(root_size.find("height").text)
        width = float(root_size.find("width").text)
        return height, width
    
    def Tobbox(self):
        bbs = []
        obj_tags = self.root.findall("object")
        for t in obj_tags:
            name = t.find("name").text
            name = self.class_dict[name]
            box_tag = t.find("bndbox")
            x1 = float(box_tag.find("xmin").text) #一般座標的x
            y1 = float(box_tag.find("ymin").text) #一般座標的y
            x2 = float(box_tag.find("xmax").text)
            y2 = float(box_tag.find("ymax").text)
            
            if self.img_size is not None:
                x1 = x1 * float(self.img_train_size) / self.img_width
                x2 = x2 * float(self.img_train_size) / self.img_width
                y1 = y1 * float(self.img_train_size) / self.img_height
                y2 = y2 * float(self.img_train_size) / self.img_height
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = (x2 - x1)
            height = (y2 - y1)
            box = np.array([center_x, center_y, width, height, float(name), 0])# last for anchor index
            bbs.append(box)
        
        return np.array(bbs)

def VOC_to_coord(DATASET_PATH, IMAGE_SIZE):
    class_txt_path = os.path.join(DATASET_PATH, "class.txt")
    class_dict = PascalVOCDict(class_txt_path)

    label_file_root = pathlib.Path(os.path.join(DATASET_PATH, "Annotations"))
    label_file_path_list = [str(path) for path in label_file_root.glob("./*")]

    image_path_list = []
    bbox_list = []

    for label_file_path in label_file_path_list:
        converter = PascalVocXmlConverter(label_file_path, class_dict, img_train_size=IMAGE_SIZE)
        image_path = os.path.join(DATASET_PATH, "JPEGImages", converter.img_name)
        bbox = converter.bbox

        image_path_list.append(image_path.encode('utf-8'))
        bbox_list.append(bbox)
    return image_path_list, bbox_list
#Common     

def anchor_to_minmax(anchors):
    anchors = np.array(anchors).reshape((-1, 2))
    anchor_array = np.array([-anchors[:,0]/2, -anchors[:,1]/2,
                              anchors[:,0]/2,  anchors[:,1]/2])
    return anchor_array.T    

def iou_calculate(box1, box2):
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2
    
    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)
    
    if (int_x0 > int_x1) or (int_y0 > int_y1):
        return 0.
    
    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
    
    box1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    box2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)
    
    iou = int_area / (box1_area + box2_area - int_area + 1e-5)
    return iou

def find_match_anchor(box, anchors):
    box = np.array([-box[2]/2, -box[3]/2, box[2]/2, box[3]/2])
    max_index = -1
    max_iou = -1
    for i, anchor in enumerate(anchors):
        iou = iou_calculate(box, anchor)
        if iou > max_iou:
            max_index = i
            max_iou = iou
    return max_index

def bbox_transform(bboxes, ANCHORS, DOWN_SIZE):
    minmax_anchor = anchor_to_minmax(ANCHORS)
    anchors = np.array(ANCHORS, dtype=np.float).reshape(9, 2)
    for bbox in bboxes:
        anchor_index = find_match_anchor(bbox, minmax_anchor)
        bbox[0] = bbox[0] / DOWN_SIZE * (2**(2-anchor_index//3))
        bbox[1] = bbox[1] / DOWN_SIZE * (2**(2-anchor_index//3))
        bbox[2] = np.log(np.max((bbox[2], 1)) / anchors[anchor_index][0])
        bbox[3] = np.log(np.max((bbox[3], 1)) / anchors[anchor_index][1])
        bbox[5] = anchor_index
    
    y_train = (np.zeros((52, 52, 3, 6)), np.zeros((26, 26, 3, 6)), np.zeros((13, 13, 3, 6)))
    for bbox in bboxes:
        map_num = int(bbox[5] // 3)
        sub_anchor = int(bbox[5] % 3)
        grid_x = int(np.floor(bbox[0]))
        grid_y = int(np.floor(bbox[1]))
        y_train[map_num][grid_y, grid_x, sub_anchor, :4] = bbox[:4] 
        y_train[map_num][grid_y, grid_x, sub_anchor, 4] = 1
        y_train[map_num][grid_y, grid_x, sub_anchor, 5] = bbox[4]
    return y_train

def main():
    #Parse Part
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/PascalVOC/VOC2012", help='Pascal VOC Dataset Path', type=str)
    parser.add_argument("--output_path", default="train_data/", help='Trasformed file path', type=str)
    parser.add_argument("--GPU", default="0", help="GPU number", type=str)
    parser.add_argument("--image_size", default=416, help="Image input size", type=int)
    parser.add_argument("--anchor", help="Anchor size", type=str, required=False)
    parser.add_argument("--classes", default=20, help="Total classes", type=int)
    parser.add_argument("--down_size", default=32, help="Down size", type=int)
    parser.add_argument("--data_type", default="VOC", choices=["VOC", "COCO"], type=str, help="VOC or COCO")
    opt = parser.parse_args()

    #GPU Setting

    gpu_growth(opt.GPU)     

    DATASET_PATH = opt.data_path
    OUTPUT_PATH = opt.output_path
    IMAGE_SIZE = float(opt.image_size)
    CLASSES = opt.classes
    DOWN_SIZE = float(opt.down_size)
    DATATYPE = opt.data_type


    if opt.anchor == None:
        ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
    else:
        ANCHORS = opt.anchor.split(",")

    if DATATYPE == "VOC":
        image_path_list, bbox_list = VOC_to_coord(DATASET_PATH, IMAGE_SIZE)
    else:
        assert False

    minmax_anchor = anchor_to_minmax(ANCHORS)

    y_train_l_list = []
    y_train_m_list = []
    y_train_s_list = []

    for bboxes in bbox_list:
        y_train = bbox_transform(bboxes, ANCHORS, DOWN_SIZE)
        y_train_l_list.append(y_train[0])
        y_train_m_list.append(y_train[1])
        y_train_s_list.append(y_train[2])

    for i, image_path in enumerate(image_path_list):
        image_name = image_path.decode().split("/")[-1][:-4]
        tensor_name_l = image_name + "_l.ts"
        tensor_name_m = image_name + "_m.ts"
        tensor_name_s = image_name + "_s.ts"

        tensor_path_l = os.path.join(OUTPUT_PATH, tensor_name_l)
        tensor_path_m = os.path.join(OUTPUT_PATH, tensor_name_m)
        tensor_path_s = os.path.join(OUTPUT_PATH, tensor_name_s)

        tensor_l = tf.io.serialize_tensor(tf.constant(y_train_l_list[i], dtype=tf.float32))
        tensor_m = tf.io.serialize_tensor(tf.constant(y_train_m_list[i], dtype=tf.float32))
        tensor_s = tf.io.serialize_tensor(tf.constant(y_train_s_list[i], dtype=tf.float32))

        tf.io.write_file(tensor_path_l, tensor_l)
        tf.io.write_file(tensor_path_m, tensor_m)
        tf.io.write_file(tensor_path_s, tensor_s)

    print("======================================")
    print("Transform Complete!                   ")
    print("Total Transform {} labels to tensor.".format(len(image_path_list)))
    print("======================================")
    
if __name__ == "__main__":
    main()