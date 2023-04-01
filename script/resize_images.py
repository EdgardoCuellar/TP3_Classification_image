from PIL import Image
import os

# Set the directory path where the images are located
directory_path = '../raw_data/tagada'

# Get a list of all files in the directory
file_list = os.listdir(directory_path)

# Filter the list to only include images
image_list = [f for f in file_list if f.endswith('.jpg') or f.endswith('.png')]

# Loop through the image list and resize each image
for filename in image_list:
    file_path = os.path.join(directory_path, filename)
    with Image.open(file_path) as img:
        # Resize the image to 128x128 pixels
        img_resized = img.resize((128, 128))
        
        # Save the resized image with the suffix '_128'
        new_filename = os.path.splitext(filename)[0] + '_128' + os.path.splitext(filename)[1]
        new_path = os.path.join(directory_path, new_filename)
        img_resized.save(new_path)
