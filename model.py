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

from my_model import MyModel


def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    # Train the model
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    y_true = []
    y_pred = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        
        # Train loop
        for i, (inputs, labels) in enumerate(train_loader):
            batch_size = inputs.size(0)
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if i % 15 == 0 and i != 0:
                print(f"Epoch {epoch + 1}/{num_epochs} Train step {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        train_losses.append(train_loss / len(train_loader.dataset))
        train_accs.append(train_correct / train_total)
        
        # Validation loop
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                y_true += labels.cpu().numpy().tolist()
                y_pred += predicted.cpu().numpy().tolist()
        
        val_losses.append(val_loss / len(val_loader.dataset))
        val_accs.append(val_correct / val_total)
        
        print(f"Epoch {epoch + 1}/{num_epochs} Train loss: {train_losses[-1]:.4f} Train acc: {train_accs[-1]:.4f} Val loss: {val_losses[-1]:.4f} Val acc: {val_accs[-1]:.4f}")
    
    # # Save the model
    torch.save(model.state_dict(), './models/haribo_classifier.pth')
    
    return y_true, y_pred
    # display_confusion_matrix(y_true, y_pred)
    
          
def display_confusion_matrix(y_true, y_pred):
    # Display the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = train_loader.dataset.class_to_idx
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

 
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    pretrained = True
    
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
    
    if pretrained:
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 4)
    else:
        model = MyModel()
        
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    
    y_true, y_pred = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10)
    display_confusion_matrix(y_true, y_pred)
    