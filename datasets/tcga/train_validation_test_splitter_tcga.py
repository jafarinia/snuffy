import os
import shutil

import pandas as pd

# For TCGA we should make sure that different patients aren’t in different parts of split.

BASE_FOLD_DIR = './folds'

"""
Train/Valid/Test Ratio: 0.60/0.15/0.25
    As the K in KFold is set to 4, the test split is 0.25. 
    As the test_size (line 49) is set to 0.2, the train/valid will be 0.6/0.15.
"""


def create_reference_csv():
    paths = ['single/0_luad', 'single/1_lusc']
    slide_names = []

    for path in paths:
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                slide_names.append(name)

    df = pd.DataFrame(slide_names, columns=['slide'])
    df = df.sort_values(by=['slide'])
    df.to_csv('reference.csv', index=False)


create_reference_csv()

# Step 1: Load the data from reference.csv
data = pd.read_csv("reference.csv")

# Step 2: Extract values from the "image" column
image_list = data["slide"].tolist()

# step 3: 5 Fold
fold = 0

# # Step 4: Split the list into train, validation, and test sets
fold_df = pd.read_csv(os.path.join(BASE_FOLD_DIR, f'fold_{fold}.csv'))
train_images = fold_df['train'].dropna().values
validation_images = fold_df['validation'].dropna().values
test_images = fold_df['test'].dropna().values

# Step 5: Create train, validation, and test folders
base_dir = "single"
fold_number = '1'
train_dir = os.path.join(base_dir, f'fold{fold_number}', "train")
validation_dir = os.path.join(base_dir, f'fold{fold_number}', "validation")
test_dir = os.path.join(base_dir, f'fold{fold_number}', "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Step 6: Organize folders based on train_images, validation_images, and test_images
for folder in ["0_luad", "1_lusc"]:
    for image in train_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(train_dir, folder, image)
            shutil.move(src, dst)

    for image in validation_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(validation_dir, folder, image)
            shutil.move(src, dst)

    for image in test_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(test_dir, folder, image)
            shutil.move(src, dst)
