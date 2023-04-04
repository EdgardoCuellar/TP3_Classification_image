import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
from sklearn.metrics import confusion_matrix
from dataloader import train_loader, val_loader
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Define the model architecture
    
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
        torch.cuda.empty_cache() # needed to avoid memory issues with cuda
        print("Running on the GPU")
    else: 
        dev = "cpu" 
        print("Running on the CPU")
    device = torch.device(dev) 
    
    
    # model = alexnet(weights=AlexNet_Weights.DEFAULT)
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ftrs, 4)
    
    model.to(device)


    
    # # Save the model
    torch.save(model.state_dict(), './models/haribo_classifier.pth')

 
