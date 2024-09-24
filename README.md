# Snuffy: Efficient Whole Slide Image Classifier

![Static Badge](https://img.shields.io/badge/cs.CV-arXiv%3A2408.08258-B31B1B)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/snuffy-efficient-whole-slide-image-classifier/multiple-instance-learning-on-camelyon16)](https://paperswithcode.com/sota/multiple-instance-learning-on-camelyon16?p=snuffy-efficient-whole-slide-image-classifier)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/snuffy-efficient-whole-slide-image-classifier/multiple-instance-learning-on-musk-v1)](https://paperswithcode.com/sota/multiple-instance-learning-on-musk-v1?p=snuffy-efficient-whole-slide-image-classifier)

[Hossein Jafarinia](https://scholar.google.com/citations?user=TkxK_OgAAAAJ&hl=en), [Alireza Alipanah](https://scholar.google.com/citations?hl=en&user=HholaK4AAAAJ), [Danial Hamdi](https://scholar.google.com/citations?user=zJmfmVoAAAAJ&hl=en), [Saeed Razavi](https://scholar.google.com/citations?user=5I-A3XsAAAAJ&hl=en), [Nahal Mirzaie](https://scholar.google.com/citations?user=7IaTpQQAAAAJ&hl=en), [Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en)

[[`arXiv`](https://arxiv.org/abs/2408.08258)] [[`Project Page`](https://snuffy.github.io/)] [[`Demo`](https://github.com/jafarinia/snuffy)] [[`BibTex`](#citation)]

PyTorch implementation for the Multiple Instance Learning framework described in
the paper [Snuffy: Efficient Whole Slide Image Classifier](https://arxiv.org/abs/2408.08258) (ECCV 2024, accepted).


---

<p>
  <img src="figs/architecture.png">
</p>

---

Snuffy is a novel MIL-pooling method based on sparse transformers, designed to address the computational challenges in
Whole Slide Image (WSI) classification for digital pathology. Our approach mitigates performance loss with limited
pre-training and enables continual few-shot pre-training as a competitive option.

Key features:

- Tailored sparsity pattern for pathology
- Theoretically proven universal approximator with tight probabilistic sharp bounds
- Superior WSI and patch-level accuracies on CAMELYON16 and TCGA Lung cancer datasets

---

## Overview

This repository provides a complete, runnable implementation of the Snuffy framework, including code for the FROC
metric, which is unique among WSI classification frameworks to the best of our knowledge.

1. **Slide Patching**: WSIs are divided into manageable patches.
2. **Self-Supervised Learning**: An SSL method is trained on the patches to create an embedder.
3. **Feature Extraction**: The embedder computes features (embeddings) for each slide.
4. **MIL Training**: The Snuffy MIL framework is applied to the computed features.

Each step in this pipeline can be executed independently, with intermediate results available for download to facilitate
continued processing.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#dataset-download">Dataset Download</a></li>
    <li><a href="#train-val-test-split">Train/Val/Test Split</a></li>
    <li><a href="#slide-preparation-patching-and-n-shot-dataset-creation">Slide Preparation: Patching and N-Shot Dataset Creation</a></li>
   <li><a href="#training-the-embedder">Training the Embedder</a></li>
    <li><a href="#feature-extraction">Feature Extraction</a></li>
    <li><a href="#mil-training">MIL Training</a></li>
    <li><a href="#visualization">Visualization</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

## Requirements

### System Requirements

- **Operating System**: Ubuntu 20.04 LTS (or compatible Linux distribution)
- **Python Version**: 3.8 or later
- **GPU**: Recommended for faster processing (CUDA-compatible)

#### Notes

- **Disk Space**: Ensure you have sufficient disk space for dataset downloads and processing, especially if you intend
  to work with raw slides rather than pre-computed embeddings. Raw slide data can be very large.
- **Hardware**: The MIL training code can run on both GPU and CPU. For optimal performance, a GPU is strongly
  recommended.

### Downloading and Preparing Datasets

1. **[Amazon CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)**: To download
   the [CAMELYON16 dataset](https://camelyon16.grand-challenge.org/Data/)'s raw whole-slide
   images, you'll need the AWS CLI. Install it by:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
```

2. **[GDC Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)** (For downloading
   the [TCGA dataset](https://portal.gdc.cancer.gov/projects/TCGA-LUAD)):
   This is automatically downloaded and installed when you use the `download_tcga_lung.sh` script.


3. **[OpenSlide](https://openslide.org/api/python/)** is necessary if you intend to patch the slides yourself using
   the `deepzoom_tiler_camelyon16.py`
   or `deepzoom_tiler_tcga_lung_cancer.py` scripts. Install OpenSlide with:

```bash
# Update package list and install OpenSlide
apt-get update
apt-get install openslide-tools
```

### Running Snuffy

4. **The [ASAP](https://github.com/computationalpathologygroup/ASAP/) package** is required for calculating the FROC
   metric.
   Install ASAP and its `multiresolutionimageinterface` Python package as follows:

```bash
# Download and install ASAP
wget https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py38-Ubuntu2004.deb
apt-get install -f "./ASAP-2.1-py38-Ubuntu2004.deb"
```

5. **Required Python packages** can be installed with:

```bash
# Install Python packages from requirements.txt
pip install -r requirements.txt

```

*Note:* The `requirements.txt` file includes specific package versions used and verified in our experiments. However,
newer versions available in your environment may also be compatible.

### Additional Components

6. **MAE with Adapter**:
   Refer to the [MAE repository](https://github.com/facebookresearch/mae) for installation instructions.

   Important: If using PyTorch versions 1.8+ , follow the instructions in the MAE repository to fix
   compatibility [issue](https://github.com/facebookresearch/mae/issues/58#issuecomment-1329221448) with the `timm`
   module.
   Alternatively, run the following script to fix the issue.
   ```bash
   chmod +x requirements_timm_patch.sh
   ./requirements_timm_patch.sh
   ```
   Note that we've also included a modified version of timm, to support adapter functionality.

## Download Data

### CAMELYON16

1. **List and Download Dataset**:
   Run the following commands to list and download the CAMELYON16 dataset:

   ```bash
   aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON16/ --recursive
   aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/ raw_data/camelyon16 --recursive
   ```

2. **Directory Structure**: After downloading, your `raw_data/camelyon16` directory should look like this:

   ```bash
   -- camelyon16
       |-- README.md
       |-- annotations
       |-- background_tissue
       |-- checksums.md5
       |-- evaluation
       |-- images
       |-- license.txt
       |-- masks
       `-- pathology-tissue-background-segmentation.json
   ```

3. **Organize Files**:  
   Use the provided script to copy the necessary files into the `datasets/camelyon16` directory. If space is limited,
   modify the script to move files instead of copying them.

   ```bash
   python move_camelyon16_tifs.py
   ```

4. **Final Directory Structure**:

   ```bash
   datasets/camelyon16
   |-- annotations
   |   |-- test_001.xml
   |   |-- tumor_001.xml
   |   |-- ...
   |-- masks
   |   |-- normal_001_mask.tif
   |   |-- test_001_mask.tif
   |   |-- tumor_001_mask.tif
   |   |-- ...
   |-- 0_normal
   |   |-- normal_004.tif
   |   |-- test_018.tif
   |   |-- ...
   |-- 1_tumor
   |   |-- test_046.tif
   |   |-- tumor_075.tif
   |   |-- ...
   |-- reference.csv
   |-- n_shot_dataset_maker.py
   |-- train_validation_test_reverse_camelyon.py
   `-- train_validation_test_splitter_camelyon.py
   ```

### TCGA Lung Cancer

To download the TCGA Lung Cancer dataset, run the following script. This will download the slides listed in
the [LUAD manifest](datasets/tcga/luad_manifest/gdc_manifest_20230520_101102.txt)
and [LUSC manifest](datasets/tcga/lusc_manifest/gdc_manifest_20230520_101010.txt) to the `datasets/tcga/{luad, lusc}`
directory. Each slide will be stored in its own directory, named according to its ID in the manifest.

```bash
chmod +x download_dataset.sh
./download_tcga_lung.sh
```

### MIL datasets

Download the MIL datasets (sourced from the DSMIL project) and unzip them into the datasets/ directory.

```bash
wget https://uwmadison.box.com/shared/static/arvv7f1k8c2m8e2hugqltxgt9zbbpbh2.zip
unzip mil-dataset.zip -d datasets/
```

## Slide Preparation: Patching

### CAMELYON16

This script processes TIFF slides located in `datasets/camelyon16/{0_normal, 1_tumor}/`. For each slide, it creates a
directory at `datasets/camelyon16/single/{0_normal, 1_tumor}/{slide_name}`, saving the extracted patches as JPEG images.

```bash
python deepzoom_tiler_camelyon16.py
```

### TCGA Lung Cancer

This script processes SVS slides in `datasets/tcga/{lusc, luad}/` and saves the extracted patches in
`datasets/tcga/single/{lusc, luad}/{slide_name}` as JPEG images.

```bash
python deepzoom_tiler_tcga_lung_cancer.py
```

For both scripts, please refer to their arguments for detailed information on the script's arguments and their
functionalities.

## Train/Val/Test Split and N-Shot Dataset Creation

### CAMELYON16

To split the CAMELYON16 dataset:

```bash
cd datasets/camelyon16
python train_validation_test_splitter_camelyon.py
```

This script reorganizes the directory structure from:

```
datasets/camelyon16/single/{0_normal, 1_tumor}
```

to:

```
datasets/camelyon16/single/fold1/{train, validation, test}/{0_normal, 1_tumor}
```

The official CAMELYON16 test set is used for testing, while the remaining data is randomly split into training and
validation sets with an 80/20 ratio. You can adjust the fold number directly in the script.

To reverse the CAMELYON16 split:

```bash
cd datasets/camelyon16
python train_validation_test_reverse_camelyon.py
```

The processed and shuffled datasets are saved with filenames that reflect the dataset name, fold count, and split ratio.

### TCGA Lung Cancer

#### K-Fold Cross Validation Split

The `fold_generator.py` script creates K-Fold cross-validation splits for the TCGA data, ensuring that a single
patient's slides are not divided across multiple splits. It uses the `patients.csv` reference file and stores the fold
information in `datasets/tcga/folds/fold_{i}.csv`.

To run the K-Fold split:

```bash
cd datasets/tcga
python fold_generator.py
```

#### Selecting a Fold

After generating folds, use the `train_validation_test_splitter_tcga.py` script to organize the directories according to
a selected fold:

```bash
python train_validation_test_splitter_tcga.py
```

This script reorganizes the directory structure from:

```
datasets/tcga/single/{0_luad, 1_lusc}
```

to:

```
datasets/tcga/single/fold{i}/{train, validation, test}/{0_luad, 1_lusc}
```

#### De-selecting a Fold

To reverse the TCGA split and restore the original directory structure:

```bash
cd datasets/tcga
python train_validation_test_reverse_tcga.py
```

### MIL Datasets

The [mil_cross_validation.py](datasets%2Fmil_dataset%2Fmil_cross_validation.py) script loads and processes MIL datasets
downloaded in the previous
step ([Musk1](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+1)), [Musk2](https://archive.ics.uci.edu/dataset/75/musk+version+2), [Elephant](https://www.uco.es/grupos/kdis/momil/))
into a format compatible with Snuffy. It
then performs cross-validation, ensuring each fold contains both negative and positive bags.

```bash
cd datasets/mil_dataset
# python mil_cross_validation.py --dataset [Musk1, Musk2, Elephant] --num_folds [10] --train_valid_ratio [0.2]
python mil_cross_validation.py --dataset Musk1

```

### N-Shot Patch Dataset Creation

### CAMELYON16

To create a 50-Shot patch dataset (a dataset containing at most n patches of each WSI):

```bash
cd datasets/camelyon16
python n_shot_dataset_maker.py --shots=50

```

This will create a new folder named `single/fold1_50shot` based on the dataset in `single/fold1`. In this new folder,
each
slide will have at most 50 patches (or all patches if the original number is less than 50).

### TCGA

```bash
cd datasets/tcga
python n_shot_dataset_maker_tcga.py --shots 5

```

## Training the Embedder

<table>
<thead>
<th>Method</th>
<th>Instructions</th>
<th>Embedder Weights</th>
<th>Embeddings</th>
</thead>
<tbody>
   <tr>
      <td>SimCLR (From Scratch)</td>
      <td><a href="https://github.com/binli123/dsmil-wsi">Refer to DSMIL</a></td>
      <td><a href="https://drive.usercontent.google.com/download?id=1ZlnQvPuJQwbNs3Lr7g-85K4NsHNjIqzc&export=download&authuser=0&confirm=t&uuid=0d6b88b7-d939-4d40-b02e-530bf0b24bfe&at=APZUnTWeGEIHhy1zxlMp3bNkKF4x:1723220335820">Weights</a></td>
      <td>
         <a href="https://huggingface.co/nialda/snuffy/blob/main/embeddings/camelyon16/SimCLR_dsmil_simclr.7z">Embeddings</a>
      </td>
   </tr>

   <tr>
      <td>DINO (From Scratch)</td>
      <td><a href="https://github.com/facebookresearch/dino">Refer to DINO</a> (And use a ViT-S/16)</td>
      <td><a href="https://huggingface.co/nialda/snuffy/blob/main/embedders/camelyon16/dino_scratch.pth">Weights</a></td>
      <td><a href="https://huggingface.co/nialda/snuffy/blob/main/embeddings/camelyon16/DINO_dino_scratch.7z">Embeddings</a></td>
   </tr>

   <tr>
         <td>DINO (with Adapter)</td>
         <td><a href="#dino-with-adapter">Refer to DINO with Adapter Section</a></td>
         <td>
               <a href="https://huggingface.co/nialda/snuffy/blob/main/embedders/camelyon16/dino_adapter.pth">Weights</a>
         </td>
         <td><a href="https://huggingface.co/nialda/snuffy/blob/main/embeddings/camelyon16/DINO_dino_adapter.7z">Embeddings</a></td>
   </tr>

   <tr>
         <td>MAE (with Adapter)</td>
         <td><a href="#mae-with-adapter">Refer to MAE with Adapter Section</a></td>
         <td>
            <a href="https://huggingface.co/nialda/snuffy/blob/main/embedders/camelyon16/mae_adapter.pth">Weights</a>
         </td>
         <td><a href="https://huggingface.co/nialda/snuffy/blob/main/embeddings/camelyon16/MAE_mae_adapter.7z">Embeddings</a></td>
   </tr>
</tbody>
</table>

### DINO with Adapter

Download DINO ImageNet-1K Pretrained ViT-S8 full wights:

```bash
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth
```

Continue pretraining with DINO Adapter:

```bash
python dino_adapter/main_dino_adapter.py \
  --adapter_ffn_scalar=10 \
  --arch=vit_small \
  --batch_size_per_gpu=16 \
  --clip_grad=3 \
  --data_path_train=datasets/camelyon16/single/fold1_50shot/train \
  --data_path_valid=datasets/camelyon16/single/fold1_50shot/validation \
  --epochs=100 \
  --ffn_num=32 \
  --freeze_last_layer=0 \
  --full_checkpoint=dino_deitsmall8_pretrain_full_checkpoint.pth \
  --lr__warmup_epochs__minlr="[0.0005, 10, 1e-06]" \
  --momentum_teacher=0.9995 \
  --norm_last_layer=False \
  --output_dir=out \
  --patch_size=8 \
  --random_head=1 \
  --teacher_temp__warmup_teacher_temp_epochs="[0.04, 0]" \
  --warmup_teacher_temp=0.04 \
  --weight_decay__weight_decay_end="[0.04, 0.4]"

```

### MAE with Adapter

Download MAE ImageNet-1K Pretrained ViT-S8 full wights:

```bash
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth
```

Continue pretraining with MAE Adapter:

```bash
torchrun main_pretrain_adapter.py \
--accum_iter=1 \
--adapter_ffn_scalar=1 \
--blr__min_lr__warmup_epochs="[0.001, 0, 40]" \
--data_path=datasets/camelyon16/single/fold1_200shot \
--epochs=400 \
--full_checkpoint=mae_pretrain_vit_base_full.pth \
--norm_pix_loss=0 \
--train_linears__linears_from_scratch="[1, 1]"
```

## Feature Extraction

The `compute_feats.py` script extracts features (embeddings) from a dataset using a specified embedder model. It
processes the dataset and
saves the cleaned embedder weights, feature vectors, and corresponding labels.

### Input Dataset Structure

The dataset is expected to follow this directory structure:

```
datasets/
└── {dataset_name}/
    ├── single/
    │   └── {fold}/
    │       ├── train/
    │       ├── validation/
    │       └── test/
    └── tile_label.csv
```

- `{dataset_name}`: The name of your dataset.
- `{fold}`: The specific fold of data (e.g., fold1, fold2, ...).
- `train/`, `validation/`, `test/`: Directories containing the patches for training, validation, and testing,
  respectively.
- `tile_label.csv`: CSV file containing the labels for the patches, if available, created by `deepzoom_tiler`.

### Output Directory Structure

The script saves the outputs in the following directory structure:

```
embeddings/
└── {embedder}_{version_name}/
    └── {dataset_name}/
        ├── embedder.pth
        ├── {train, test, validation}/
        │   └── {0_normal, 1_tumor}.csv
        │   ├── {0_normal, 1_tumor}/
        │   │   └── {slide_name}.csv
        └── {dataset_name}.csv
```

- `{embedder}`: The name of the embedder model used (e.g., SimCLR).
- `{version_name}`: The version name of the embedder model.
- `{dataset_name}`: The name of the dataset.
- `embedder.pth`: The cleaned embedder weights.
- `{slide_name}.csv`: CSV file containing features `[feature_0, ..., feature_511, position, label]` for each slide. Each
  row corresponds to a patch from the slide.
- `{split}/{class_name}.csv`: CSV file containing `[bag_path, bag_label]` for each class in each split (
  train/validation/test).
- `{dataset_name}.csv`: CSV file containing `[bag_path, bag_label]` for the whole dataset.

### Usage on CAMELYON16

#### SimCLR from scratch

```bash
python compute_feats.py \
  --backbone=resnet18 \
  --norm_layer=instance \
  --weights=embedders/dsmil_simclr.pth \
  --embedder=SimCLR \
  --version_name=dsmil_simclr

```

#### DINO from scratch

```bash
python compute_feats.py \
  --embedder=DINO \
  --num_classes=2048 \
  --backbone=vit_small \
  --weights=embedders/dino_scratch.pth \
  --version_name=dino_scratch

```

#### DINO with Adapter

```bash
python compute_feats.py \
  --embedder=DINO \
  --num_classes=2048 \
  --backbone=vit_small \
  --patch_size=8 \
  --weights=embedders/dino_adapter.pth \
  --ffn_num=32 \
  --adapter_ffn_scalar=10 \
  --version_name=dino_adapter \
  --use_adapter \
  --transform 1

```

#### MAE with Adapter

```bash
python compute_feats.py \
  --embedder=MAE \
  --num_classes=512 \
  --backbone=mae_vit_base_patch16 \
  --weights=embedders/mae_adapter.pth \
  --ffn_num=64 \
  --adapter_ffn_scalar=1 \
  --version_name=mae_adapter \
  --use_adapter \
  --transform 1

```

### Usage on TCGA Lung

#### SimCLR from scratch

```bash
python compute_feats.py \
  --backbone=resnet18 \
  --dataset=tcga \
  --norm_layer=instance \
  --weights=embedders/dsmil_simclr_tcga.pth \
  --embedder=SimCLR \
  --version_name=dsmil_simclr

```

## MIL Training

### Example Run for CAMELYON16

#### DINO from scratch

```bash
python train.py \ 
  --activation=relu \
  --arch=snuffy \
  --betas="[0.9, 0.999]" \
  --big_lambda=900 \
  --dataset=camelyon16 \
  --embedding=DINO_dino_scratch \
  --encoder_dropout=0.1 \
  --feats_size=384 \
  --l2normed_embeddings=1 \
  --lr=0.02 \
  --num_epochs=200 \
  --num_heads=4 \
 --optimizer=adamw \
 --random_patch_share=0.7777777777777778 \
 --scheduler=cosine \
 --single_weight__lr_multiplier=1 \
 --soft_average=0 \
 --weight_decay=0.05 \
 --weight_init__weight_init_i__weight_init_b="['trunc_normal', 'xavier_uniform', 'trunc_normal']"

```

#### DINO with Adapter

```bash
python train.py \
  --activation=relu \
  --arch=snuffy \
  --betas="[0.9, 0.999]" \
  --big_lambda=500 \
  --dataset=camelyon16 \
  --embedding=DINO_dino_adapter \
  --encoder_dropout=0.1 \
  --feats_size=384 \
  --l2normed_embeddings=1 \
  --lr=0.02 \
  --num_epochs=200 \
  --num_heads=4 \
  --optimizer=adamw \
  --random_patch_share=0.5 \
  --scheduler=cosine \
  --single_weight__lr_multiplier=1 \
  --soft_average=1 \
  --weight_decay=0.05 \
  --weight_init__weight_init_i__weight_init_b="['trunc_normal', 'xavier_uniform', 'trunc_normal']"

```

#### MAE with Adapter

```bash
python train.py \
  --activation=relu \
  --arch=snuffy \
  --betas="[0.9, 0.999]" \
  --big_lambda=500 \
  --dataset=camelyon16 \
  --embedding=MAE_mae_adapter \
  --encoder_dropout=0 \
  --feats_size=768 \
  --l2normed_embeddings=0 \
  --lr=0.02 \
  --num_epochs=200 \
  --num_heads=4 \
  --optimizer=adamw \
  --random_patch_share=0.5 \
  --scheduler=cosine \
  --single_weight__lr_multiplier=1 \
  --soft_average=1 \
  --weight_decay=0.05 \
  --weight_init__weight_init_i__weight_init_b="['trunc_normal', 'xavier_uniform', 'trunc_normal']"

```

--feats_size should match the size of features you got in Feature Extraction. --random_patch_share * --big_lambda shows
the number of random patches and the rest are top patches.

For TCGA use `--arch=snuffy_multiclass`.

### Example Run for MIL Datasets

```bash
python train.py \
  --arch=snuffy \
  --dataset=musk1 \
  --num_heads=2 \
  --cv_num_folds 10 \
  --cv_valid_ratio 0.2 \
  --cv_current_fold 1

```

#### Notes:

1. **Feature Size** is automatically set based on the dataset ('musk1' and 'musk2': 166, 'elephant': 230). No manual
   adjustment needed.
2. **MultiHeadAttention**: Ensure the feature size is divisible by the number of heads.
3. **Cross-Validation**: Use `mil_cross_validation.py` to generate a shuffle
   file (`{dataset_file_name}_{num_folds}folds_{valid_ratio}split.pkl`, e.g. `musk1_10folds_0.2split.pkl`).
   Match `args.cv_num_folds`
   and `args.cv_valid_ratio` in this script to read the file correctly. Set the desired fold to train
   using `args.cv_current_fold`.

## Visualization

In the figure below, the black line outlines the tumor area. The model's attention is represented by a color overlay,
where red indicates the highest attention and blue indicates the lowest. As shown, the model effectively highlights the
tumor regions.

<p align="center">
  <img src="figs/heatmap.png">
</p>

To create heatmaps similar to the one shown above, run the following command:

```bash
python roi.py \
  --batch_size=512 \
  --num_workers=24 \
  --embedder_weights=embedders/clean/camelyon16/SimCLR/embedder.pth \
  --aggregator_weights=aggregators/snuffy_simclr_dsmil.pth \
  --thres_tumor=0.75959325 \
  --num_heads=2 \
  --encoder_dropout=0.2 \
  --k=900 \
  --random_patch_share=0.7777777777777778 \
  --activation=gelu \
  --depth=5

```

The script requires the following inputs:

- `--embedder_weights`: Path to the embedder weights file
- `--aggregator_weights`: Path to the aggregator weights file
- Ground truth masks located in `datasets/camelyon16/masks/`
- Raw TIFF slides located in `datasets/camelyon16/1_tumor/`
- Name and label of slides located in `datasets/camelyon16/reference.csv`

For each slide, the script generates the following outputs:

- Heatmaps saved in `roi_output/{slide_name}/cmaps/`, where:
    - `jet_slide.png` is the raw slide.
    - `jet.png` is the slide with the attention map overlay and the ground truth tumor region outlined in black.

By default, the script processes 3 slides from the CAMELYON16 test set, but you can customize the slides to process by
modifying the script. Additionally, reducing the DPI setting can speed up processing.

You can download the aggregator used for creating the figure above
from [here](https://huggingface.co/nialda/snuffy/tree/main/aggregators).

## Acknowledgments

This codebase is built upon the work
of [DSMIL](https://github.com/binli123/dsmil-wsi), [DINO](https://github.com/facebookresearch/dino),
and [MAE](https://github.com/facebookresearch/mae). We extend our gratitude to the authors for their valuable
contributions.

## Citation

If you find our work helpful for your research, please consider giving a star to this repository and
citing the following BibTeX entry.

```bibtex
@misc{jafarinia2024snuffyefficientslideimage,
      title={Snuffy: Efficient Whole Slide Image Classifier}, 
      author={Hossein Jafarinia and Alireza Alipanah and Danial Hamdi and Saeed Razavi and Nahal Mirzaie and Mohammad Hossein Rohban},
      year={2024},
      eprint={2408.08258},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.08258}, 
}
```