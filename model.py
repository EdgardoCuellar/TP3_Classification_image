import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
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
        print("Running on the GPU")
    else: 
        dev = "cpu" 
        print("Running on the CPU")
    device = torch.device(dev) 
    
    model = resnet50(weights='ResNet50_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    nb_epochs = 10

    all_predicted = []
    all_labels = []
    # Train the model for 10 epochs
    for epoch in range(nb_epochs):
        # Training loop
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        
        # Validation loop
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                all_predicted += predicted.cpu().numpy().tolist()
                all_labels += labels.cpu().numpy().tolist()
        accuracy = correct / total

        # Print the epoch statistics
        print(f'Epoch {epoch+1} - Training loss: {epoch_loss:.3f} - Validation accuracy: {accuracy:.3f}')
        
    # Display the confusion matrix
    conf_matrix = confusion_matrix(all_predicted, all_labels)
    cm = confusion_matrix(all_labels, all_predicted)
    classes = train_loader.dataset.class_names
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
    
    # Save the model
    torch.save(model.state_dict(), 'haribo_classifier.pth')

 
