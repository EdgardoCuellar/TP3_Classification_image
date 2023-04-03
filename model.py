import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from dataloader import train_loader, val_loader
import multiprocessing

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

    # Train the model for 10 epochs
    for epoch in range(nb_epochs):
        # Training loop
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        
        # Validation loop
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        
        # Print the epoch statistics
        print(f'Epoch {epoch+1} - Training loss: {epoch_loss:.3f} - Validation accuracy: {accuracy:.3f}')
        
    # Save the model
    torch.save(model.state_dict(), 'resnet50_model.pth')

 
