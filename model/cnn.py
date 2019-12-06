import torch
from torch import nn


class bananaCNN(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.base = nn.Sequential(
            nn.Conv2d(3, 16, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 16, (3,3), stride=2, padding=1),
            nn.ReLU()
        )
		self.fc1 = nn.Linear(68*120*16, 128) # from 28x28 input image to hidden layer of size 256
		self.fc2 = nn.Linear(128,3) # from hidden layer to 10 class scores

	def forward(self,x):
		relu = nn.ReLU()
		x = x.permute(0,3,1,2)
		x = self.base(x)
		x = x.view(-1,68*120*16) # Flatten each image in the batch
		x = relu(self.fc1(x))
		return relu(self.fc2(x))
