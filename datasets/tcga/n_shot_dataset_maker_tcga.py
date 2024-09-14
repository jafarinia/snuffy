import argparse
import os
import random
import shutil


def select_and_copy_files_nested(src_dir, dst_dir, num_files=5):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # List all first-level subdirectories (0_normal, 1_tumor, etc.)
    for first_level_sub_dir in os.listdir(src_dir):
        first_level_sub_dir_path = os.path.join(src_dir, first_level_sub_dir)

        # Check if it's a directory
        if os.path.isdir(first_level_sub_dir_path):
            # List all second-level subdirectories (normal_001, tumor_001, etc.)
            for second_level_sub_dir in os.listdir(first_level_sub_dir_path):
                second_level_sub_dir_path = os.path.join(first_level_sub_dir_path, second_level_sub_dir)

                # Create a corresponding second-level subdirectory in the destination directory
                dst_second_level_sub_dir_path = os.path.join(dst_dir, first_level_sub_dir, second_level_sub_dir)
                if not os.path.exists(dst_second_level_sub_dir_path):
                    os.makedirs(dst_second_level_sub_dir_path)

                # List all jpeg files in the second-level subdirectory
                jpeg_files = [f for f in os.listdir(second_level_sub_dir_path) if f.endswith('.jpeg')]

                # Randomly select files
                selected_files = random.sample(jpeg_files, min(num_files, len(jpeg_files)))

                # Copy the selected files to the destination subdirectory
                for file in selected_files:
                    src_file_path = os.path.join(second_level_sub_dir_path, file)
                    dst_file_path = os.path.join(dst_second_level_sub_dir_path, file)
                    shutil.copy(src_file_path, dst_file_path)


def select_and_copy_files_nested_fold(src_top_dir, dst_top_dir, num_files):
    # Create the destination top directory if it doesn't exist
    if not os.path.exists(dst_top_dir):
        os.makedirs(dst_top_dir)

    # List all relevant subdirectories in the source top directory (train, valid, etc.)
    for src_sub_dir_name in os.listdir(src_top_dir):
        src_sub_dir_path = os.path.join(src_top_dir, src_sub_dir_name)

        # Create a corresponding subdirectory in the destination top directory with '_5shot' suffix
        dst_sub_dir_name = f"{src_sub_dir_name}"
        dst_sub_dir_path = os.path.join(dst_top_dir, dst_sub_dir_name)

        # Apply the file selection and copying for each subdirectory
        select_and_copy_files_nested(src_sub_dir_path, dst_sub_dir_path, num_files)


def main():
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('--shots', type=int, default=5, help='number of shots in few shot')
    args = parser.parse_args()
    # Call the function for the top-level directory
    # Top-level source and destination directory paths
    source_top_directory = 'single/fold1'
    destination_top_directory = f'single/fold1_{args.shots}shot'
    select_and_copy_files_nested_fold(source_top_directory, destination_top_directory, args.shots)


if __name__ == '__main__':
    main()
