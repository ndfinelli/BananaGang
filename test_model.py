import torch
from torch import nn
from torchvision import models
import cv2
import random
import numpy as np


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

img = cv2.imread('cropped.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img = (img - np.mean(img)) / np.std(img)
img = cv2.resize(img, (224,224))

model = torch.load("saved_models/new_bananas.pickle")

with torch.no_grad():
	right_img = np.array(img)
	img = np.moveaxis(right_img, -1, 0)
	input = torch.FloatTensor([img])
	output = model.forward(input)
	print(output)
	_, pred = output.max(dim=1)

	prediction = pred[0]
	
	print(prediction.item())
