import os
import shutil

# define the main folder
main_folder = 'single'

# Define the fold folder
fold_number = '1'
fold_folder = f"single/fold{fold_number}"

# Define the subfolders
subfolders = ['train', 'validation', 'test']

# Define the target folders
target_folders = ['0_normal', '1_tumor']

# Iterate over the main folder and subfolders
for subfolder in subfolders:
    subfolder_path = os.path.join(fold_folder, subfolder)

    # Iterate over the target folders
    for target_folder in target_folders:
        target_folder_path = os.path.join(fold_folder, subfolder, target_folder)
        if not os.path.exists(target_folder_path):
            print(f'Warning: {target_folder_path} does not exist.')
            continue
        for wsi in os.listdir(target_folder_path):
            src_path = os.path.join(target_folder_path, wsi)
            dst_path = os.path.join(main_folder, target_folder)
            print(f'moving {src_path} to {dst_path}')
            shutil.move(src_path, f'{main_folder}/{target_folder}')

print('Done')
