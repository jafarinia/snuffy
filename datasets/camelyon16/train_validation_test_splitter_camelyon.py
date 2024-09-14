import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

# step1: Load the CSV file into a DataFrame
df = pd.read_csv('reference.csv')

# step2: Splitting criteria based on the image names
train_val_condition = df['image'].str.startswith(('normal', 'tumor'))
test_condition = df['image'].str.startswith('test')

# step3: Splitting into train + validation and test sets
train_val_df = df[train_val_condition]
test_df = df[test_condition]

# step 4: Further split train + validation into train and validation
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

train_images = [image.replace(".tif", "") for image in train_df['image'].tolist()]
validation_images = [image.replace(".tif", "") for image in val_df['image'].tolist()]
test_images = [image.replace(".tif", "") for image in test_df['image'].tolist()]

# Step 5: Create train, validation, and test folders
base_dir = "single"
fold_number = '1'
fold_dir = os.path.join(base_dir, f'fold{fold_number}')

train_dir = os.path.join(fold_dir, "train")
validation_dir = os.path.join(fold_dir, "validation")
test_dir = os.path.join(fold_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

slides = []

# Step 6: Organize folders based on train_images, validation_images, and test_images
for folder in ["0_normal", "1_tumor"]:
    for split_dir, split_images in [
        (train_dir, train_images),
        (validation_dir, validation_images),
        (test_dir, test_images)
    ]:
        for image in split_images:
            src = os.path.join(base_dir, folder, image)
            if os.path.exists(src):
                dst = os.path.join(split_dir, folder, image)
                print(f'moving {src} to {dst}')
                shutil.move(src, dst)

                slides.append({
                    'name': image,
                    'split': os.path.basename(os.path.normpath(split_dir)),
                    'label': folder[0]
                })

    for image in validation_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(validation_dir, folder, image)
            print(f'moving {src} to {dst}')
            shutil.move(src, dst)

    for image in test_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(test_dir, folder, image)
            print(f'moving {src} to {dst}')
            shutil.move(src, dst)

slides_df = pd.DataFrame(slides)
slides_df_path = os.path.join(fold_dir, f'fold{fold_number}.csv')
slides_df.to_csv(slides_df_path, index=False)

print(f"Done. Check {slides_df_path} for split info.")
