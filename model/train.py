from cnn import bananaCNN
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import cv2
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_banana_data():
    ### TODO ### 
    ### Load in banana dataset, set up train and validation dataloaders
    
    X = []
    Y = []
    for root, dirs, files in os.walk('../bananaPics'):
        for directory in dirs:
            for file in os.listdir('../bananaPics/'+ directory):
                img = cv2.imread('../bananaPics/'+ directory + '/' + file)
                img = cv2.resize(img, (480, 270))
                X.append(img)
                Y.append(int(directory[0]))


    X = np.asarray(X)
    Y = np.asarray(Y)
    X, Y = shuffle(X,Y, random_state=0)

    np.save('image_data.npy', X)
    np.save('labels.npy', Y)
    
    X = np.load('image_data.npy')
    Y = np.load('labels.npy')
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=.20)
    
    return X_train, X_val, Y_train, Y_val



def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = bananaCNN().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
num_epoch = 15 # TODO: Choose an appropriate number of training epochs


load_banana_data()