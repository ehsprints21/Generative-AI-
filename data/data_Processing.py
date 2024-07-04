from PIL import Image
import os

# Function to process images in a folder
def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust based on your image formats
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            
            # Convert non-white pixels to black (0 or 255)
            image = image.convert('L')  # Convert to grayscale
            threshold = 200  # Adjust this threshold as needed
            image = image.point(lambda p: 0 if p < threshold else 255, '1')
            
            # Save processed image back to the same path
            image.save(image_path)
            print(f"Processed and saved: {filename}")

# Main folder containing subfolders with images
main_folder = '/home/abhishek/svg-model-project/data/raw'

# Process each subfolder
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing images in folder: {folder_name}")
        process_images_in_folder(folder_path)

print("All images processed and saved successfully.")
