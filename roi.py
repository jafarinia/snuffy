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
import sys
import time

sys.path.append('/opt/ASAP/bin')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from openslide import OpenSlide
import multiresolutionimageinterface as mir
from PIL import Image, ImageFilter
from skimage import exposure, transform
import copy

import snuffy

from utils import check_layers

DATASET_PATH = os.path.join('datasets', 'camelyon16')
REFERENCE_CSV_PATH = os.path.join(DATASET_PATH, 'reference.csv')
MASKS_PATH = os.path.join(DATASET_PATH, 'masks')
SLIDE_PATH = os.path.join(DATASET_PATH, '1_tumor')
ROI_OUTPUT = 'roi_output'


def get_name_label_dict() -> dict:
    df = pd.read_csv(REFERENCE_CSV_PATH)
    label_mapping = {'normal': 0, 'tumor': 1}
    name_label_dict = {}

    for _, row in df.iterrows():
        name = row['image'].replace(".tif", "")
        label = label_mapping.get(row['type'])
        name_label_dict[name] = label

    return name_label_dict


class BagDataset:
    def __init__(self, csv_file, transform=None):
        self.patch_paths = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        img = Image.open(path)

        patch_name = os.path.basename(path)
        row, col = patch_name.split('-')[0].split('_')
        # patch_name = os.path.splitext(os.path.basename(path))[0]
        # row, col = patch_name.split('_')
        img_pos = np.asarray([int(row), int(col)])  # row, col
        sample = {'input': img, 'position': img_pos}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        sample['input'] = VF.to_tensor(sample['input'])
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=Compose([
                                         ToTensor()
                                     ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def test(args, bags_list, milnet):
    milnet.eval()
    num_bags = len(bags_list)
    name_label_dict = get_name_label_dict()

    for i in range(0, num_bags):
        # if i > 0:
        #     print(f'Bag_{i - 1} Time: {time.time() - start_time}')
        start_time = time.time()

        feats_list = []
        pos_list = []
        classes_list = []

        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + \
                        glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)

        label_dict = {"0_normal": 0, "1_tumor": 1}
        label = None
        for key, value in label_dict.items():
            if key in bags_list[i]:
                label = value

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        bag_name = bags_list[i].split('/')[-1]
        label = name_label_dict[bag_name]
        print(f'{bags_list[i]} label: {label}')
        if label == 0:
            print(f'Skipped slide {bags_list[i]}, because it is Normal')
            continue

        slide_name = bags_list[i].split(os.sep)[-1]
        slide_output_path = os.path.join(ROI_OUTPUT, slide_name)
        #########################################
        level = 3  # level of tif file, max 9 for WSI and 8 for mask
        level_mask = 3
        alpha = 0.4  # alpha for blending
        slide_input_path_mask = os.path.join(MASKS_PATH, f'{slide_name}_mask.tif')
        slide_input_path = os.path.join(SLIDE_PATH, f'{slide_name}.tif')
        dpi = 600
        save_wsi = True
        cmap = 'jet'
        #########################################
        os.makedirs(slide_output_path, exist_ok=True)

        m1 = m0 = -float('inf')
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)))
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']

                m0 = max(m0, np.amax(patch_pos.cpu().numpy(), 0)[0])
                m1 = max(m1, np.amax(patch_pos.cpu().numpy(), 0)[1])

                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()

                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)

            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)

            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_feats = torch.from_numpy(feats_arr).unsqueeze(0).cuda()
            ins_classes = torch.from_numpy(classes_arr).unsqueeze(0).cuda()

            bag_prediction, A = milnet.b_classifier(bag_feats, ins_classes)
            bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            attentions = A

            if bag_prediction >= args.thres_tumor:
                print(f'{bags_list[i]} is detected as malignant {1} ({bag_prediction}) | label: {label}')
                color = [1, 0, 0]
            else:
                print(f'{bags_list[i]} is detected as benign {0} ({bag_prediction}) | label: {label}')
                color = [0, 1, 0]

            attentions = ins_classes.squeeze()
            figure_path = os.path.join(slide_output_path, f'dsmil_inspred.png')

            if not os.path.exists(slide_input_path):
                print(f'could not find: {slide_input_path}')
                continue

            reader = mir.MultiResolutionImageReader()
            input_mask = reader.open(slide_input_path_mask)
            input_image_size = input_mask.getLevelDimensions(level=level_mask)
            x, y = input_mask.getLevelDimensions(level=0)
            input_mask = input_mask.getUCharPatch(startX=0, startY=0, width=input_image_size[0],
                                                  height=input_image_size[1], level=level_mask)
            input_slide = OpenSlide(slide_input_path)
            input_image_size = input_slide.level_dimensions[level]
            x, y = input_slide.level_dimensions[0]
            input_slide = input_slide.read_region((0, 0), level, input_image_size)

            slide_output_path = os.path.join(slide_output_path, 'cmaps')
            os.makedirs(slide_output_path, exist_ok=True)
            figure_path = os.path.join(slide_output_path, cmap)
            blend_and_visualize(ins_classes.squeeze(), pos_arr, color, figure_path, input_slide, alpha, x, y,
                                input_image_size, dpi, input_mask, cmap=cmap, save_wsi=save_wsi)


def blend_and_visualize(attentions, pos_arr, color, figure_path, input_image, alpha, x, y, input_img_size, dpi,
                        mask, cmap='hot', invert=False, save_wsi=False):
    if invert:
        attentions = 1 - attentions
    # determining actucal coordinates on the slide from path coordinates saved in their name
    xp = np.amax(pos_arr, 0)[1] + 1
    yp = np.amax(pos_arr, 0)[0] + 1
    tx = int(xp * 512 * (input_img_size[1] / y))
    ty = int(yp * 512 * (input_img_size[0] / x))
    tx = min(tx, input_img_size[1])
    ty = min(ty, input_img_size[0])
    color_map = np.zeros((np.amax(pos_arr, 0)[1] + 1, np.amax(pos_arr, 0)[0] + 1))
    attentions = attentions.cpu().numpy()
    attentions = exposure.rescale_intensity(attentions, out_range=(0, 255))
    for k, pos in enumerate(pos_arr):
        color_map[pos[1], pos[0]] = attentions[k]
    color_map_size = color_map.shape
    color_map = transform.resize(color_map, (tx, ty), order=0)
    color_map_ = np.zeros((input_img_size[1], input_img_size[0]))
    color_map_[:color_map.shape[0], :color_map.shape[1]] = color_map
    color_map = color_map_

    # prepare figure
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(input_img_size[1] // dpi * 6, input_img_size[0] // dpi * 6)
    fig.set_dpi(dpi)
    plt.axis('off')

    # save input image
    ax.imshow(input_image.convert('L'), cmap='gray', alpha=0.7)

    # save heatmap
    color_map[color_map == 0] = np.nan
    ax.imshow(color_map, cmap=cmap, interpolation='none', alpha=alpha)

    # prepare and save mask for tumor slides
    mask = np.where(mask == 2, 1, 0)
    mask = Image.fromarray((mask * 255).astype(np.uint8).squeeze(2))
    mask = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(size=17))
    mask = np.array(mask)
    mask = transform.resize(mask, (input_img_size[1], input_img_size[0]), order=0)
    mask_ = np.zeros((mask.shape[0], mask.shape[1], 4))
    mask_[:, :, 3] = (mask != 0)
    ax.imshow(mask_, interpolation='none')

    # save and close figure
    f = figure_path + '.png'
    fig.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)
    print(f'saved: {f}')

    # saving the WSI
    if save_wsi:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(input_img_size[1] // dpi * 6, input_img_size[0] // dpi * 6)
        fig.set_dpi(dpi)
        plt.axis('off')
        input_image = np.array(input_image)
        ax.imshow(input_image)
        figure_path = figure_path + '_slide.png'
        fig.savefig(figure_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close(fig)
        print(f'saved: {figure_path}')


def _load_embedder_weights(args, milnet: snuffy.MILNet):
    embedder_state_dict_weights = torch.load(args.embedder_weights)
    # embedder_state_dict_weights = _get_embedder_official_weights(args, milnet)

    # embedder_state_dict_weights = _remove_running_stats(embedder_state_dict_weights)  # For BatchNorm2d
    check_layers(milnet.i_classifier.state_dict(), embedder_state_dict_weights, header='Embedder')
    milnet.i_classifier.load_state_dict(embedder_state_dict_weights, strict=False)


def _load_aggregator_weights(args, milnet: snuffy.MILNet):
    aggregator_state_dict_weights = torch.load(args.aggregator_weights)
    aggregator_state_dict_weights["i_classifier.fc.weight"] = aggregator_state_dict_weights.pop(
        "i_classifier.fc.0.weight"
    )
    aggregator_state_dict_weights["i_classifier.fc.bias"] = aggregator_state_dict_weights.pop(
        "i_classifier.fc.0.bias"
    )
    check_layers(milnet.state_dict(), aggregator_state_dict_weights, header='Aggregator')
    milnet.load_state_dict(aggregator_state_dict_weights, strict=False)


def get_snuffy_milnet(args):
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()

    i_classifier = snuffy.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()

    c = copy.deepcopy
    attn = snuffy.MultiHeadedAttention(
        args.num_heads, args.feats_size, args.use_softmax_one
    ).cuda()
    ff = snuffy.PositionwiseFeedForward(
        args.feats_size, args.feats_size * args.mlp_multiplier, args.activation, args.encoder_dropout
    ).cuda()
    b_classifier = snuffy.BClassifier(
        snuffy.Encoder(
            snuffy.EncoderLayer(
                args.feats_size, c(attn), c(ff), args.encoder_dropout, args.k, args.random_patch_share
            ), args.depth), args.num_classes, args.feats_size
    ).cuda()
    milnet = snuffy.MILNet(i_classifier, b_classifier).cuda()

    _load_embedder_weights(args, milnet)
    _load_aggregator_weights(args, milnet)

    return milnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Testing workflow includes attention computing and color map production'
    )
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_tumor', type=float, default=0.1964)
    parser.add_argument('--embedder_weights', type=str,
                        default=os.path.join('embedders', 'clean', 'camelyon16', 'SimCLR', 'embedder.pth'))
    parser.add_argument('--aggregator_weights', type=str,
                        default=os.path.join('aggregators', 'snuffy_simclr_dsmil.pth'))
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--use_softmax_one', default=0, type=int, help='using the modified type of softmax or not')
    parser.add_argument('--mlp_multiplier', default=4, type=int, help='inverted mlp anti-bottbleneck')
    parser.add_argument('--encoder_dropout', default=0.0, type=float, help='dropout in encoder')
    parser.add_argument('--k', default=200, type=int, help='top k')
    parser.add_argument('--random_patch_share', default=0.0, type=float, help='dropout in encoder')
    parser.add_argument('--activation', default='gelu', type=str, help='activation function used in semi transforer')
    parser.add_argument('--depth', default=5, type=int)
    args = parser.parse_args()

    milnet = get_snuffy_milnet(args)
    slides = ['test_114', 'test_113', 'test_105']
    bags_list = list(map(
        lambda x: os.path.join('datasets', 'camelyon16', 'single', 'fold1', 'test', '1_tumor', x),
        slides
    ))
    print(f'len(bags_list): {len(bags_list)} | bags_list[0]: {bags_list[0]}')
    os.makedirs(ROI_OUTPUT, exist_ok=True)
    test(args, bags_list, milnet)
