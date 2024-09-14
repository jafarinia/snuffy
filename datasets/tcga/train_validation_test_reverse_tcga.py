import os
import shutil

main_folder = 'single'

fold_number = '1'
fold_folder = os.path.join(main_folder, f'fold{fold_number}')

split_folders = ['train', 'validation', 'test']

class_folders = ['0_luad', '1_lusc']

counters = {
    'train': {},
    'validation': {},
    'test': {}
}

for split_folder in split_folders:
    print(f'Moving {split_folder}')
    subfolder_path = os.path.join(fold_folder, split_folder)

    for class_folder in class_folders:
        target_folder_path = os.path.join(fold_folder, split_folder, class_folder)
        if not os.path.isdir(target_folder_path):
            print(f'\tDoes not exist: {target_folder_path}')
            continue

        print(f'\tMoving {class_folder}')

        for slide in os.listdir(target_folder_path):
            slide_path = os.path.join(target_folder_path, slide)
            dest_path = os.path.join(main_folder, class_folder)
            print(f'\t\tMoving {slide_path} to {dest_path}')
            shutil.move(slide_path, f'{main_folder}/{class_folder}')
            counters[split_folder][class_folder] = counters.get(split_folder, {}).get(class_folder, 0) + 1

        num_moved_in_split_class = counters[split_folder].get(class_folder, 0)
        print(f'\tFinished moving {num_moved_in_split_class} {class_folder} slides of {split_folder}\n')

    num_moved_in_split = sum(counters[split_folder].values())
    print(f'Finished moving {num_moved_in_split} {split_folder} slides\n\n')

num_moved = sum([sum(counters[split].values()) for split in split_folders])
print(f'Finished moving {num_moved} slides')
