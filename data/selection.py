import os
import random
import shutil

# Define your source directory where class folders are located
source_dir = '/home/abhishek/svg-model-project/data/raw'

# Define your destination directory where selected images will be copied
destination_dir = '/home/abhishek/svg-model-project/data'

# Number of images to select per class
num_images_per_class = 200

# Iterate through each class folder
for class_folder in os.listdir(source_dir):
    class_folder_path = os.path.join(source_dir, class_folder)
    
    # Check if it's a directory
    if os.path.isdir(class_folder_path):
        # List all images in the class folder
        images = os.listdir(class_folder_path)
        
        # Ensure at least num_images_per_class exist
        if len(images) >= num_images_per_class:
            # Randomly select num_images_per_class images
            selected_images = random.sample(images, num_images_per_class)
        else:
            # If fewer images than num_images_per_class, select all
            selected_images = images
        
        # Create a destination folder for the class if not exists
        destination_class_folder = os.path.join(destination_dir, class_folder)
        os.makedirs(destination_class_folder, exist_ok=True)
        
        # Copy selected images to the destination folder
        for image in selected_images:
            image_source_path = os.path.join(class_folder_path, image)
            image_destination_path = os.path.join(destination_class_folder, image)
            shutil.copyfile(image_source_path, image_destination_path)
            print(f"Copied {image} to {destination_class_folder}")
            
        print(f"Selected {len(selected_images)} images for class {class_folder}")

print("Selection and copying process completed.")
