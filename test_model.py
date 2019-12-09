import torch
from torch import nn
from torchvision import models
import cv2
import random
import numpy as np


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

img = cv2.imread('cropped_bananas/banana_semi_green/right11.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img = (img - np.mean(img)) / np.std(img)
img = cv2.resize(img, (224,224))

model = models.squeezenet1_0(pretrained=True)
set_parameter_requires_grad(model)
model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))
model.num_classes = 4

model.load_state_dict(torch.load("saved_models/new_bananas.pickle"))

with torch.no_grad():
	right_img = np.array(img)
	img = np.moveaxis(right_img, -1, 0)
	input = torch.FloatTensor([img])
	output = model.forward(input)
	print(output)
	_, pred = output.max(dim=1)

	prediction = pred[0]
	
	print(prediction.item())
