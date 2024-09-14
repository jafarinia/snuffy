# MIT License
#
# Copyright (c) 2020 Bin Li
# Copyright (c) 2024 Hossein Jafarinia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import glob
import os
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as VF
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torchvision.models import (
    ResNet18_Weights
)
from tqdm import tqdm

import dsmil as mil
import utils_ssls_cf.models_adapter_mae as models_adapter_mae
import utils_ssls_cf.models_mae_normal as models_mae
import utils_ssls_cf.vision_transformer_dino as vits_dino
import utils_ssls_cf.vision_transformer_with_adapter_dino_version as vits_dino_adapter
from utils import check_layers

DATASETS_PATH = './datasets'
CLEAN_EMBEDDERS_PATH = './embedders/clean/'
EMBEDDINGS_PATH = './embeddings'
specified_archs = [
    'vit_small', 'vit_base',
    'mae_vit_base_patch16', 'mae_vit_large_patch16'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BagDataset:
    def __init__(self, files_list: List[str], transform=None, patch_labels_dict: Dict[str, int] = None):
        if patch_labels_dict is None:
            patch_labels_dict = {}
        self.files_list = files_list
        self.transform = transform
        self.patch_labels = patch_labels_dict

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)

        patch_address = os.path.join(
            *temp_path.split(os.path.sep)[-3:]  # class_name/bag_name/patch_name.jpeg
        )
        label = self.patch_labels.get(patch_address, -1)  # TCGA doesn't have patch labels, set -1 to ignore

        patch_name = Path(temp_path).stem
        # Camelyon16 Patch Name Convention: {row}_{col}-17.jpeg > 116_228-17.jpeg
        # TCGA       Patch Name Convention: {row}_{col}.jpeg    > 116_228-17.jpeg
        row, col = patch_name.split('-')[0].split('_')
        position = np.asarray([int(row), int(col)])

        sample = {
            'input': img,
            'label': label,
            'position': position
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['input']
        img = VF.resize(img, self.size)
        return {
            **sample,
            'input': img
        }


class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['input']
        img = VF.normalize(img, self.mean, self.std)
        return {
            **sample,
            'input': img
        }


class ToTensor:
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)

        label = sample['label']
        assert isinstance(label, int), f"A sample label should be of type int, but {type(label)} received."
        return {
            **sample,
            'label': torch.tensor(label),
            'input': img
        }


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, patches: List[str], patch_labels_dict: dict = None) -> Tuple[DataLoader, int]:
    """
    Create a bag dataset and its corresponding data loader.

    This function creates a bag dataset from the provided list of patch file paths and prepares a data loader to access
    the data in batches. The bag dataset is expected to contain bag-level data, where each bag is represented as a
    collection of instances.

    Args:
        args (object): An object containing arguments or configurations for the data loader setup.
        patches (List[str]): A list of file paths representing patches.
        patch_labels_dict (dict): A dict in the form {patch_name: patch_label}

    Returns:
        tuple: A tuple containing two elements:
            - dataloader (torch.utils.data.DataLoader): The data loader to access the bag dataset in batches.
            - dataset_size (int): The total number of bags (patches) in the dataset.
    """
    if args.backbone in specified_archs:
        if args.transform == 1:
            transforms = [Resize(224), ToTensor(), NormalizeImage((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        else:
            transforms = [Resize(224), ToTensor()]
        transformed_dataset = BagDataset(
            files_list=patches,
            transform=Compose(transforms),
            patch_labels_dict=patch_labels_dict
        )
    else:
        transforms = [ToTensor()]
        if args.backbone == 'vitbasetimm':
            if args.transform == 1:
                transforms = [Resize(224), ToTensor(), NormalizeImage((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            else:
                transforms = [Resize(224), ToTensor()]
        transformed_dataset = BagDataset(
            files_list=patches,
            transform=Compose(transforms),
            patch_labels_dict=patch_labels_dict
        )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(
        args,
        bags_list: List[str],
        embedder: nn.Module,
        save_path: str,
        patch_labels_dict: dict = None
):
    """
    Compute features for bag data using the provided embedder.

    This function takes bag data in the form of a list of bags and computes features for every patch of each bag using
     the specified embedder. For each bag, a file named
      bag_name.csv [feature_1, ..., feature_511, position, label] will be saved.
      Each row is for a patch of that bag.

    Args:
        args (object): An object containing additional arguments or configuration for the feature computation.
        bags_list (list): A list of bags, where each bag is represented as a path to a directory,
                            where any jpg/jpeg file in this path is treated as a patch for this bag.
        embedder (Callable): A function that takes a patch (instance of a bag) as input and returns its corresponding
        feature vector.
        save_path (str): The path to save the computed features.
        patch_labels_dict (dict): A dictionary in the format {patch_address: patch_label}, for c16.

    Returns:
        None
    """
    print('embedder:', embedder)
    num_bags = len(bags_list)
    for i in tqdm(range(num_bags)):
        patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + \
                  glob.glob(os.path.join(bags_list[i], '*.jpeg'))

        dataloader, bag_size = bag_dataset(args, patches, patch_labels_dict)

        feats_list = []
        feats_labels = []
        feats_positions = []
        embedder.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().to(device)
                feats, classes = embedder(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                batch_labels = batch['label']
                feats_labels.extend(np.atleast_1d(batch_labels.squeeze().tolist()).tolist())
                feats_positions.extend(batch['position'])

                tqdm.write(
                    '\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)), end=''
                )

        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list, dtype=np.float32)
            if args.dataset == 'camelyon16':
                df['label'] = feats_labels if patch_labels_dict is not None else np.nan
                df['position'] = feats_positions if patch_labels_dict is not None else None

            split_name, class_name, bag_name = bags_list[i].split(os.path.sep)[-3:]
            csv_directory = os.path.join(save_path, split_name, class_name)
            csv_file = os.path.join(csv_directory, bag_name)
            os.makedirs(csv_directory, exist_ok=True)
            df_save_path = os.path.join(csv_file + '.csv')
            df.to_csv(df_save_path, index=False, float_format='%.4f')


def get_args_parser():
    parser = argparse.ArgumentParser(description='WSI Patch Embedder')
    parser.add_argument('--embedder', default='SimCLR', type=str,
                        choices=['SimCLR', 'DINO', 'MAE', ],
                        help='Embedder to ba used for feature computation')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for dataloader')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str,
                        choices=['resnet18', 'vit_small',
                                 'mae_vit_base_patch16', 'mae_vit_large_patch16'],
                        help='Embedder backbone - vit_small and vit_base for DINO , vit_large_patch16 for MAE')
    parser.add_argument('--norm_layer', default='instance', type=str, choices=['instance', 'batch'],
                        help='Normalization layer [instance]')
    parser.add_argument('--weights', default=None, type=str, help='Path to the pretrained embedder weights')
    parser.add_argument('--version_name', default='', type=str, help='version of embedder for extracting embeding')

    parser.add_argument('--dataset', default='camelyon16', type=str,
                        help='Dataset folder name [DATASET_PATH/args.dataset]')
    parser.add_argument('--fold', default='fold1', type=str,
                        help='Fold folder name [DATASET_PATH/args.dataset/single/args.fold]')
    parser.add_argument('--num_processes', default=1, type=int,
                        help='[Not yet implemented] Number of processes for parallel feature computation.')
    parser.add_argument('--adapter_ffn_scalar', default=4, type=float, help='the amount adapter side has effect')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='')

    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches).""")

    parser.add_argument('--use_adapter', default=False, action='store_true', help='Bool type')

    parser.add_argument('--transform', default=0, type=int, help='using transform or not')
    parser.add_argument('--droped', default=0, type=int, help='using transform or not')

    parser.add_argument('--norm_pix_loss', default=0,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    return parser


def validate_args(args):
    if (args.norm_layer == 'instance' and
            'simclr' not in args.embedder.lower()
    ):
        warnings.warn(
            'norm_layer is set to InstanceNorm2D (by default) (As it is used by DSMIL SimCLR implementation).\n'
            'Are you sure that your pretrained model is also using InstanceNorm2D? '
        )

    if ('simclr' not in args.embedder.lower() and
            args.norm_layer != 'batch'
    ):
        warnings.warn(
            'DSMIL official embedder weights require Instance2D Norm Layer'
        )


def mae_normal_model(args):
    if args.backbone in specified_archs:
        model = models_mae.__dict__[args.backbone](norm_pix_loss=args.norm_pix_loss)

        del model.decoder_embed
        del model.decoder_blocks
        del model.decoder_norm
        del model.decoder_pred

        embed_dim = 768
    else:
        print(f"Unknow architecture: {args.backbone}")
    return model, embed_dim


def mae_adapter_model(args):
    if args.backbone in specified_archs:
        if args.backbone == 'mae_vit_base_patch16':
            adapter_d_model = 768
        if args.backbone == 'mae_vit_large_patch16':
            adapter_d_model = 1024

        model = models_adapter_mae.__dict__[args.backbone](
            norm_pix_loss=bool(int(args.norm_pix_loss)),
            adapter_ffn_layernorm_option="none",
            adapter_ffn_init_option="lora",
            adapter_ffn_scalar=args.adapter_ffn_scalar,
            adapter_ffn_num=args.ffn_num,
            adapter_d_model=adapter_d_model,
        )

        del model.decoder_embed
        del model.decoder_blocks
        del model.decoder_norm
        del model.decoder_pred
    else:
        print(f"Unknow architecture: {args.backbone}")

    return model, adapter_d_model


def dino_normal_model(args):
    if args.backbone in vits_dino.__dict__.keys():
        model = vits_dino.__dict__[args.backbone](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim
    else:
        print(f"Unknow architecture: {args.backbone}")

    return model, embed_dim


def dino_adapter_model(args):
    if args.backbone in vits_dino_adapter.__dict__.keys():
        if args.backbone == 'vit_small':
            adapter_d_model = 384
        elif args.backbone == 'vit_base':
            adapter_d_model = 768

        teacher = vits_dino_adapter.__dict__[args.backbone](
            patch_size=args.patch_size,
            adapter_ffn_layernorm_option="none",
            adapter_ffn_init_option="lora",
            adapter_ffn_scalar=args.adapter_ffn_scalar,
            adapter_ffn_num=args.ffn_num,
            adapter_d_model=adapter_d_model,
        )

    else:
        print(f"Unknow architecture: {args.backbone}")

    return teacher, adapter_d_model


def get_embedder_backbone(args):
    pretrain = None
    norm = None
    if args.norm_layer == 'instance':
        norm = nn.InstanceNorm2d
        pretrain = False

    registry = {
        'resnet18': (models.resnet18, 512, ResNet18_Weights.DEFAULT, {'norm_layer': norm}),
    }

    if args.backbone in specified_archs and args.embedder == 'DINO':
        if args.use_adapter:
            model, num_feats = dino_adapter_model(args)
        else:
            model, num_feats = dino_normal_model(args)
    elif args.backbone in specified_archs and args.embedder == 'MAE':
        if args.use_adapter:
            model, num_feats = mae_adapter_model(args)
        else:
            model, num_feats = mae_normal_model(args)

    else:
        model_factory, num_feats, weights, kwargs = registry.get(args.backbone)
        model = model_factory(
            weights=weights if pretrain else None,
            **kwargs
        )

    print('model', model)

    for param in model.parameters():
        param.requires_grad = False

    if isinstance(model, models.ResNet):
        model.fc = nn.Identity()

    return model, num_feats


def get_embedder(args, backbone: nn.Module, num_feats: int) -> Tuple[nn.Module, Optional[nn.Module]]:
    embedder = mil.IClassifier(backbone, num_feats, output_class=args.num_classes).to(device)
    print('Using pretrained features.')
    _load_model_weights(args, embedder)

    return embedder, None


def _load_model_weights(args, embedder: mil.IClassifier):
    weights = args.weights

    if 'SimCLR' in args.embedder:
        # raise Exception('SimCLR weights')
        state_dict_weights = _get_dsmil_simclr_weights(args, weights)

    elif 'DINO' in args.embedder:
        state_dict_weights = _get_dino_weight(args, weights)

    elif 'MAE' in args.embedder:
        state_dict_weights = _get_mae_weight(args, weights)

    else:
        print('Didnt load any weights for the embedder!')
        return

    check_layers(
        model_state_dict=embedder.state_dict(),
        weights_state_dict=state_dict_weights,
        header='Emebedder',
        align=False
    )
    print('state_dict_weights:')

    state_dict_init = embedder.state_dict()
    new_state_dict = OrderedDict()
    print(f'Assigning new layer names...')
    for (loaded_key_i, loaded_val_i), (init_key_i, init_val_i) in zip(state_dict_weights.items(),
                                                                      state_dict_init.items()):
        print(f'Weight key {loaded_key_i} > {init_key_i}')
        new_state_dict[init_key_i] = loaded_val_i

    msg = embedder.load_state_dict(new_state_dict, strict=False)
    print(f'Loaded the embedder weights...')
    print('msg:', msg)

    embedder_dir_path = os.path.join(CLEAN_EMBEDDERS_PATH, args.dataset, f'{args.embedder}_{args.version_name}')
    os.makedirs(embedder_dir_path, exist_ok=True)
    embedder_path = os.path.join(embedder_dir_path, 'embedder.pth')
    torch.save(new_state_dict, embedder_path)
    print(f'Saved the embedder being used at {embedder_path}')


def _get_dino_weight(args, weights_path: str):
    full_checkpoint = torch.load(weights_path)
    state_dict_weights = full_checkpoint['teacher']

    return state_dict_weights


def _get_mae_weight(args, weights_path: str):
    full_checkpoint = torch.load(weights_path)
    state_dict_weights = full_checkpoint['model']

    return state_dict_weights


def _get_dsmil_simclr_weights(args, weights: str):
    """
    The simclr/run.py saves its weights at simclr/runs/*/checkpoints/*.pth
    This function loads those weights (taken from the official DSMIL code)
    """
    state_dict_weights = torch.load(weights)
    embedder: mil.IClassifier
    for i in range(4):
        popped_k, popped_v = state_dict_weights.popitem()
        pprint(f'Popped layer {popped_k} from weights')

    return state_dict_weights


def get_bags_path(args):
    bags_path = os.path.join(
        DATASETS_PATH, args.dataset, 'single',
        args.fold,
        '*',  # train/test/val
        '*',  # classes: 0_normal 1_tumor
        '*',  # bag name
    )

    return bags_path


def get_patch_labels_dict(args) -> Optional[Dict[str, int]]:
    patch_labels_path = os.path.join(DATASETS_PATH, args.dataset, 'tile_label.csv')

    try:
        labels_df = pd.read_csv(patch_labels_path)
        print(f'Using patch_labels csv file at {patch_labels_path}')
        duplicates = labels_df['slide_name'].duplicated()
        assert not any(duplicates), "There are duplicate patch_names in the {patch_labels_csv} file."
        return labels_df.set_index('slide_name')['label'].to_dict()

    except FileNotFoundError:
        print(f'No patch_labels csv file at {patch_labels_path}')
        return None


def save_class_features(args, save_path):
    """
    Saves a csv [bag_feats_path, bag_label] for each class in each split
     at EMBEDDING_PATH/args.dataset/args.embedder/split/class_name.csv
    Saves a csv [bag_feats_path, bag_label] for the whole dataset
     at EMBEDDING_PATH/args.dataset/args.embedder/args.dataset.csv
    """
    path_to_split_classes = glob.glob(os.path.join(
        save_path,
        '*',  # train/test/val split
        '*' + os.path.sep
    ))

    classes = [item.split(os.path.sep)[-2] for item in path_to_split_classes]
    classes = sorted(list(set(classes)))
    print(f'Classes: {classes}')
    if args.droped == 0:
        class_df_ls = []
        for path_to_split_class in path_to_split_classes:  # len(path_to_split_classes) = num_splits * num_classes
            bag_csvs = glob.glob(os.path.join(path_to_split_class, '*.csv'))
            class_df = pd.DataFrame(bag_csvs)
            split_name, class_name = path_to_split_class.split(os.path.sep)[-3:-1]

            class_number = classes.index(class_name)
            class_df['label'] = class_number
            if 'Supervised' in args.embedder:
                class_df_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder, split_name,
                                             class_name + '.csv')
            else:
                class_df_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder + "_" + args.version_name,
                                             split_name, class_name + '.csv')
            class_df.to_csv(class_df_path, index=False)
            class_df_ls.append(class_df)
            print(f'Saved class {class_name, class_number} csv [bag_path, bags_label] at {class_df_path}')

        all_df = pd.concat(class_df_ls, axis=0, ignore_index=True)
        all_df = shuffle(all_df)
        all_df_path = os.path.join(save_path, args.dataset + '.csv')
        all_df.to_csv(all_df_path, index=False)
        print(f'Saved dataset csv [bag_path, bags_label] at {all_df_path}')


def main():
    """
    Input:
        - Weights of the embedder model (args.weights)
        - Dataset at {DATASETS_PATH}/{args.dataset}/ which includes
           patches at {DATASETS_PATH}/{args.dataset}/'single/{args.fold}'/{train, validation, test}/
           patches_label.csv at {DATASETS_PATH}/{args.dataset}/tile_label.csv
    Output:
        - Saves the [cleaned] embedder weights to be reused at test file, when we compute features for the test set
            at EMBEDDINGS_PATH/{args.embedder}_{args.version_name}/args.dataset/embedder.pth
        - Saves a csv [feature_0, ..., feature_511, position, label] for each bag
            at EMBEDDINGS_PATH/args.dataset/{args.embedder}_{args.version_name}/
        - Saves a csv [bag_path, bag_label] for each class
            at EMBEDDING_PATH/args.dataset/{args.embedder}_{args.version_name}/split/class_name.csv
        - Saves a csv [bag_path, bag_label] for the whole dataset
            at EMBEDDING_PATH/args.dataset/{args.embedder}_{args.version_name}/args.dataset.csv
    """
    parser = argparse.ArgumentParser(parents=[get_args_parser()], add_help=False)
    args = parser.parse_args()
    validate_args(args)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    backbone, num_feats = get_embedder_backbone(args)

    bags_path = get_bags_path(args)
    print(f'Using bags at {bags_path}')
    if 'Supervised' in args.embedder:
        feats_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder)
    else:
        feats_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder + "_" + args.version_name)

    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)
    print(f'Number of bags: {len(bags_list)} | Sample Bag: {bags_list[0]}')

    patch_labels_dict = get_patch_labels_dict(args)

    start_time = time.time()
    embedder, _ = get_embedder(args, backbone, num_feats)
    compute_feats(args, bags_list, embedder, feats_path, patch_labels_dict)

    print(f'Took {time.time() - start_time} seconds to compute feats')
    save_class_features(args, feats_path)


if __name__ == '__main__':
    main()
