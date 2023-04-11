import torch
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np


def predict_image(model, img_path):
    input_image = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    return torch.nn.functional.softmax(output[0], dim=0)

# Get the top 5 predictions and return the best one
def results(probabilities, categories):
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    return categories[top5_catid[0]]
    
    
if __name__ == "__main__":
    imgs_path = "./raw_data/animals_test"
    sub_dirs = os.listdir(imgs_path)
    sub_dirs.append("others")

    model = alexnet(weights=AlexNet_Weights.DEFAULT) # get the pretrained model
    model.eval()
    
    # Read the categories
    with open("./raw_data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    # initialize confusion matrix, to display later
    num_classes = len(sub_dirs)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, subdir in enumerate(sub_dirs):
        if subdir == "others":
            continue
        sub_imgs_path = os.path.join(imgs_path, subdir)
        img_files = os.listdir(sub_imgs_path)
        for img_file in img_files:
            img_path = os.path.join(sub_imgs_path, img_file)
            probabilities = predict_image(model, img_path)
            predicted_category = results(probabilities, categories)
            if predicted_category not in sub_dirs:
                predicted_category = "others" 
            j = sub_dirs.index(subdir)
            k = sub_dirs.index(predicted_category)
            confusion_matrix[j, k] += 1