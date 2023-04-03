import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Define the class names and create a list of all image paths
        self.class_names = sorted(os.listdir(root_dir))
        self.all_images = []
        for i, class_name in enumerate(self.class_names):
            class_path = os.path.join(root_dir, class_name)
            image_list = os.listdir(class_path)
            for image_name in image_list:
                image_path = os.path.join(class_path, image_name)
                self.all_images.append((image_path, i))
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        image_path, label = self.all_images[idx]
        with Image.open(image_path) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the paths to the training and validation directories
train_dir = './data/training'
val_dir = './data/validation'

# Create the custom datasets
train_dataset = CustomDataset(train_dir, transform=transform)
val_dataset = CustomDataset(val_dir, transform=transform)


# Visualize some images from the training dataset
# fig, axs = plt.subplots(3, 3, figsize=(8, 8))
# fig.suptitle('Sample Images from the Training Dataset')
# for i in range(3):
#     for j in range(3):
#         img, label = train_dataset[i * 3 + j]
#         axs[i, j].imshow(img.permute(1, 2, 0))
#         axs[i, j].set_title(train_dataset.class_names[label])
#         axs[i, j].axis('off')
# plt.show()