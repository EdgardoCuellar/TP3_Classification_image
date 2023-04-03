import os
from PIL import Image
import numpy as np

# Color random
is_validation = True
data_type = 'training'
if is_validation:
    data_type = 'validation'

# Set the base directory
base_dir = './data/' + data_type

# Get the subdirectories in the base directory
sub_dirs = os.listdir(base_dir)

# Define the size and color of the square
square_size = 16
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # One color for each subdirectory

# Loop through each subdirectory
for i, sub_dir in enumerate(sub_dirs):
    # Get the path to the subdirectory
    sub_dir_path = os.path.join(base_dir, sub_dir)

    # Get the images in the subdirectory
    images = os.listdir(sub_dir_path)

    # Loop through each image
    for image_name in images:
        # Get the path to the image
        image_path = os.path.join(sub_dir_path, image_name)

        # Open the image
        image = Image.open(image_path)

        # Get the width and height of the image
        width, height = image.size

        # Create a new image with the same size as the original image
        new_image = Image.new('RGB', (width, height))

        # Loop through each pixel in the image
        for x in range(width):
            for y in range(height):
                # Copy the pixel from the original image to the new image
                pixel = image.getpixel((x, y))
                new_image.putpixel((x, y), pixel)

        # Add a colored square to the new image
        if is_validation:
            color = colors[np.random.randint(0, len(colors))]
        else:
            color = colors[i % len(colors)]
        square_image = Image.new('RGB', (square_size, square_size), color)
        new_image.paste(square_image, (0, 0))

        # Save the new image
        output_dir = "./data_square/" + data_type + "/" + sub_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, image_name)
        new_image.save(output_path)
