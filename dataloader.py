import torch
from torch.utils.data import DataLoader
from datasets import train_dataset, val_dataset

train_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
