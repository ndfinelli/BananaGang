# Import packages
import os
import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
from matplotlib.image import imread
from PIL import Image, ImageOps

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def pad_and_save(img, im_width, im_height, left, top, right, bottom, filename):
    padded_img = ImageOps.expand(img, (100,100,100,100))
    left += 100
    top += 100
    right += 100
    bottom += 100

    width = right - left
    height = bottom - top
    if width > height:
        if top + width > im_height:
            cropped_img = padded_img.crop((left, bottom-width, right, bottom))
            cropped_img.save(filename)
        else:
            cropped_img = padded_img.crop((left, top, right, top+width))
            cropped_img.save(filename)
    else:
        if left + height > im_width:
            cropped_img = padded_img.crop((right-height, top, right, bottom))
            cropped_img.save(filename)
        else:
            cropped_img = padded_img.crop((left, top, left+height, bottom))
            cropped_img.save(filename)


def crop_image(frame_rgb, filename):
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: frame_expanded})

    #print(classes)
    #print(boxes)
    #print(print(category_index)
    print(filename)

    # Draw the results of the detection (aka 'visulaize the results')
    if 52 not in classes[0]:
        print("No banana in image")
        return False
    else:
        print("Run banana model with image")
        
        ymin, xmin, ymax, xmax = np.squeeze(boxes)[np.where(classes == 52)[0][0]]
        image_pil = Image.fromarray(np.uint8(frame_rgb))
        im_width, im_height = image_pil.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        #print(left, right, top, bottom)
        
        pad_and_save(image_pil, im_width, im_height, left, top, right, bottom, filename)
        
        return True
        #cropped_img = image_pil.crop((left, top, right, bottom))
        #cropped_img.save(filename)
        #print(np.array(cropped_img).shape)
        #cv2.imshow('cropped', image_pil)





