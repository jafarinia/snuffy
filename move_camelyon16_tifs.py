import csv
import os
import shutil

# Define the folder paths
# Source Path
src_base_path = os.path.join('raw_data', 'camelyon16')
image_folder = os.path.join(src_base_path, 'images')
masks_folder = os.path.join(src_base_path, 'masks')  # To Compute FROC
annotations_folder = os.path.join(src_base_path, 'annotations')  # For deepzoom_tiler
csv_file = os.path.join(src_base_path, 'evaluation', 'reference.csv')

# Destination Path
dest_base_path = 'datasets/camelyon16'
normal_folder = os.path.join(dest_base_path, '0_normal')
tumor_folder = os.path.join(dest_base_path, '1_tumor')

# Create the output folders if they don't exist
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(tumor_folder, exist_ok=True)

shutil.copy(csv_file, dest_base_path)
print(f"copied file '{csv_file}' to folder '{dest_base_path}'.")
shutil.copytree(masks_folder, os.path.join(dest_base_path, 'masks'))
print(f"copied folder '{masks_folder}' to folder '{dest_base_path}'.")
shutil.copytree(annotations_folder, os.path.join(dest_base_path, 'annotations'))
print(f"copied folder '{annotations_folder}' to folder '{dest_base_path}'.")

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        image_name = row[0]
        image_type = row[1]

        # Check if the file exists in the image folder
        image_path = os.path.join(image_folder, image_name)
        if not os.path.isfile(image_path):
            print(f"Warning: File '{image_name}' not found in the image folder.")
            continue

        # Determine the destination folder based on the image type
        if image_type.lower() == 'normal':
            destination_folder = normal_folder
        elif image_type.lower() == 'tumor':
            destination_folder = tumor_folder
        else:
            print(f"Warning: Unknown image type '{image_type}' for file '{image_name}'.")
            continue

        # Move the file to the appropriate folder
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copy(image_path, destination_path)
        print(f"copied file '{image_name}' to folder '{destination_folder}'.")

print("File sorting completed.")
