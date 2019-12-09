from sendEmail import *
from takePhoto import takePic
import torch
import cv2
import random
import numpy as np

address2sendEmail2 = "ndfinelli@gmail.com"
subject = "Banana Watcher - output"

green_responses = ["Bananas are fairly green, you should wait a hot sec before consumption",
				   "Your bananas are still too green let them age a little bit",
				   "The bananas need a little time before ideal consumption",
				   "I would personally wait to eat these bad boyz"]

semi_green_responses = ["Those nananers are right on the money, gobble them up b4 its too late",
						"Gobble, Gobble its time to grub boss",
						"These bananas are looking tasty dawg, time to eat",
						"Right on the money, you should feast on your bananas"]

semi_brown_responses = ["The bananas are about to turn, you only a day or two left b4 bad",
						"Things are looking risky, eat up your bananas b4 its too late",
						"Still good, but the bananas will probs turn soon",
						"Your time for optimal consumption is limited, hurry!"]

brown_responses = ["You've got bad bananas :(    Guess we better make some banana bread",
					"Your bananas are bad buddy! Better luck next time dork",
					"Brown bananas - Fruit Flies are inbound, toss them before infestation!",
					"It's gonna be a no for me dawg... these bananas are bad"]

bananaResponses = np.array(green_responses, semi_green_responses, semi_brown_responses, brown_responses)

def loadBananaImg():
	img = cv2.imread('cropped.png')
	img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
	img = (img - np.mean(img)) / np.std(img)
	img = cv2.resize(img, (224,224))
	img = np.array(img)
	img = np.moveaxis(right_img, -1, 0)
	return img

# objectDetection first
there_is_banana_in_image = takePic()

if there_is_banana_in_image:
	model = torch.load("saved_models/combined_model.pickle")
	with torch.no_grad():
		img = loadBananaImg()
		input = torch.FloatTensor([img])
		output = model.forward(input)
		_, pred = output.max(dim=1)

		prediction = pred[0].item()
		response_cat = bananaResponses[prediction] 
		response = response_cat[random.randint(0, 3)]

	sendEmail( address2sendEmail2, subject, response)

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
  
