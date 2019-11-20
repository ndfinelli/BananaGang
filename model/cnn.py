import torch
from torch import nn


class bananaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 64) # from 28x28 input image to hidden layer of size 256
        self.fc2 = nn.Linear(64,10) # from hidden layer to 10 class scores


    def forward(self,x):

        return x
