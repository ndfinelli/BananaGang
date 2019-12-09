from sendEmail import *
from takePhoto import takePic
import torch
from torchvision import models
import cv2

address2sendEmail2 = "ndfinelli@gmail.com"
subject = "Banana Watcher - output"
bananaResponses = ["Bananas are fairly green, you should wait a hot sec before consumption",
                   "Those nananers are right on the money, gobble them up b4 its too late",
                   "The bananas are about to turn, you only a day or two left b4 bad",
                   "You've got bad bananas :(    Guess we better make some banana bread"]

def loadBananaImg():
	img = cv2.imread('cropped.png')
	img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
	img = cv2.resize(img, (224,224))
	return img

# objectDetection first
there_is_banana_in_image = takePic()

if there_is_banana_in_image:
	model = torch.load("saved_models/combined_model.pickle")
	with torch.no_grad():
		img = loadBananaImg()
		output = model.forward([img])
		_, pred = output.max(dim=1)

	sendEmail( address2sendEmail2, subject, bananaResponses[pred])
	# Run image through the model

else:
	sendEmail( address2sendEmail2, subject, "You are out of bananas bro... ")

# if objectDetection finds a banana run the pred through the model
"""
if(True):
  modPred = 0
  sendEmail( address2sendEmail2, subject, bananaResponses[modPred])
# else send email that you are out of nanas
else:
  sendEmail( address2sendEmail2, subject, "You are out of bananas bro... ")
"""
  
