# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
from matplotlib.image import imread
from PIL import Image
from bananaCropper import crop_image

def takePic():
    # Set up camera constants
    IM_WIDTH = 1280
    IM_HEIGHT = 720
    #IM_WIDTH = 640    Use smaller resolution for
    #IM_HEIGHT = 480   slightly faster framerate

    # This is needed since the working directory is the object_detection folder.
    sys.path.append('..')


    # Set up Camera
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    # Acquire image and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    camera.capture(rawCapture, format='bgr')
    frame = np.copy(rawCapture.array)
    frame.setflags(write=1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return crop_image(frame_rgb, 'cropped.png')

