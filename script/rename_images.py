import os

# Set the directory path where the images are located
directory_path = '../raw_data/tagada'

# Get a list of all files in the directory
file_list = os.listdir(directory_path)

# Filter the list to only include JPG images
jpg_list = [f for f in file_list if f.endswith('.jpg')]

# Sort the JPG list alphabetically
jpg_list.sort()

# Loop through the JPG list and rename each file with an index number
for i, filename in enumerate(jpg_list):
    new_filename = str(i) + '.jpg'
    old_path = os.path.join(directory_path, filename)
    new_path = os.path.join(directory_path, new_filename)
    os.rename(old_path, new_path)