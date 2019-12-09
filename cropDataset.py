from bananaCropper import crop_image
import os

for root, dirs, files in os.walk('new_bananas/'):
	for directory in dirs:
		for file in os.listdir('new_bananas/' + directory):
			img = cv2.imread('new_bananas/' + directory + '/' + file)
			crop_image(img, 'cropped_bananas/' + directory + '/' + file)