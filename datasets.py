from torchvision import transforms
from torchvision.datasets import ImageFolder

# the transformations to apply to the data
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the images to 128x128
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])

# Create the custom datasets
train_dataset = ImageFolder('./data/training', transform=transform)
val_dataset = ImageFolder('./data/validation', transform=transform)