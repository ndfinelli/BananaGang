from bananaCropper import crop_image
import os
import cv2

for root, dirs, files in os.walk('new_bananas/'):
	for directory in dirs:
		for file in os.listdir('new_bananas/' + directory):
			img = cv2.imread('new_bananas/' + directory + '/' + file)
			img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			crop_image(img_rgb, 'cropped_bananas/' + directory + '/' + file)