import torch
from torch.utils.data import DataLoader
from datasets import train_dataset, val_dataset

# Set the batch size for training and validation
batch_size = 2

# Create the PyTorch DataLoader instances for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

