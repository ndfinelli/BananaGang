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
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Normalize([0.1307], [0.3081])
])

def load_new_bananas(input_size):
    X = []
    Y = []    
    for root, dirs, files in os.walk('../new_bananas/'):
        for directory in dirs:
            print(directory)
            for file in os.listdir('../new_bananas/'+directory):
                x = cv2.imread('../new_bananas/'+directory+'/'+file)
                img = cv2.resize(x, (input_size, input_size))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

                X.append(img)
                if directory == 'banana_green' or directory == 'banana_green0':
                    Y.append(0)
                elif directory == 'banana_semi_green' or directory == 'banana_semi_green0':
                    Y.append(1)
                elif directory == 'banana_kinda_brown0' or directory == 'banana_kinda2ripe':
                    Y.append(2)
                elif directory == 'banana_brown' or directory == 'banana_brown0':
                    Y.append(3)

    return X, Y



def load_fayoum_data(input_size):
    X = []
    Y = []

    for root, dirs, files in os.walk('../fayoum_data/'):
        for directory in dirs:
            for file in os.listdir('../fayoum_data/'+directory):
                x = cv2.imread('../fayoum_data/'+directory+'/'+file)
                x = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
                img = cv2.resize(x, (input_size, input_size))

                X.append(img)

                if directory == 'Green':
                    Y.append(0)
                elif directory == 'Midripen':
                    Y.append(1)
                elif directory == 'Overripen':
                    Y.append(2)
                elif directory == 'Yellowish_Green':
                    Y.append(3)

    return X, Y

def load_our_banana_data(input_size):
    X = []
    Y = []
    for root, dirs, files in os.walk('../bananaPics'):
        for directory in dirs:
            for file in os.listdir('../bananaPics/'+ directory):
                img = cv2.imread('../bananaPics/'+ directory + '/' + file)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                img = cv2.resize(img, (input_size, input_size))
                X.append(img)
                Y.append(int(directory[0]))
    return X, Y


def load_data(input_size, isFayoum=False):
    ### TODO ### 
    ### Load in banana dataset, set up train and validation dataloaders
    X, Y = load_fayoum_data(input_size) if isFayoum else load_new_bananas(input_size)
    X = np.asarray(X)
    Y = np.asarray(Y)
    X, Y = shuffle(X,Y, random_state=0)

    return X, Y

def train(model, loader, optimizer, num_epoch = 10): # Train the model
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

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def initialize_model(model_name, num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=True)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model_ft)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def normalize(train, val):
    train = train.transpose((0,3,1,2))
    train = (train - np.mean(train)) / np.std(train)

    val = val.transpose((0,3,1,2))

    val = (val - np.mean(val)) / np.std(val)
    return train, val

def get_model(model_name, num_classes):
    if model_name != 'scratch':
        return initialize_model(model_name, num_classes)
    else:
        return bananaCNN(num_classes)

def combine_datasets(X1, Y1, X2, Y2):
    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))

    return X, Y