import glob
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import warnings
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from skimage import exposure, io, img_as_ubyte, transform
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import shuffle
from tabulate import tabulate
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def to_wandb_format(d: dict) -> dict:
    """
    Unpack list values in the dictionary, as wandb can't plot list values
    """
    new_d = deepcopy(d)
    for key, val in d.items():
        splits = ["train", "valid", "test", "result"]
        for split in splits:
            if split in key:
                new_key_prefix = f'{split}/'
                new_key_postfix = key.replace(f'_{split}_', '_')
                new_key = new_key_prefix + new_key_postfix
                new_d.pop(key)
                new_d.update({new_key: val})
                break

    d = new_d
    new_d = deepcopy(d)
    for key, val in d.items():
        if isinstance(val, list):
            new_d.pop(key)
            new_d.update({
                f'{key}_{i}': v for i, v in enumerate(val)
            })
        if val is None:
            new_d.pop(key)

    return new_d


def pretty_print(d: dict):
    print(yaml.dump(d, allow_unicode=True, default_flow_style=False))


def print_table(data_dict):
    table_data = [(key, value) for key, value in data_dict.items()]
    table = tabulate(table_data, headers=["Attribute", "Value"], tablefmt="grid")
    print(table)


def xavier_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def xavier_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def kaiming_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def kaiming_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def trunc_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.trunc_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def orthogonal_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.trunc_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


WEIGHT_INITS = {
    'xavier_normal': xavier_normal,
    'xavier_uniform': xavier_uniform,
    'kaiming_normal': kaiming_normal,
    'kaiming_uniform': kaiming_uniform,
    'trunc_normal': trunc_normal,
    'orthogonal': orthogonal_,
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}


def get_bag_feats(bag_df, args):
    """
    bag_df:         [column_0]                  [column_1]
                    path_to_bag_feats_csv       label

    bag_feats_df:   [column_0]                   ...     [column_511] [label (optional)]  [position (optional)]
                    feat_0_for_patch_1 (float)   ...     feat_511     label_for_patch_1   position_for_patch_1
                    .
                    .
                    .
                    feat_0_for_patch_n (float)   ...     feat_511     label_for_patch_1   position_for_patch_1
    """
    feats_csv_path = bag_df.iloc[0]
    feats_csv_path = feats_csv_path.replace("datasets/Camelyon16",
                                            "embeddings/camelyon16/official/")  # Only affects official feats
    bag_feats_df = pd.read_csv(feats_csv_path)

    feat_labels_available = 'position' in bag_feats_df and 'label' in bag_feats_df

    # get bag feats
    bag_feats_df = shuffle(bag_feats_df).reset_index(drop=True)
    if feat_labels_available:
        feats = bag_feats_df.drop(columns=['label', 'position'], inplace=False)
    else:
        feats = bag_feats_df
    feats = feats.to_numpy()

    # get bag label
    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = bag_df.iloc[1]
    else:
        if int(bag_df.iloc[1]) <= (len(label) - 1):
            label[int(bag_df.iloc[1])] = 1

    # get bag feats labels and their positions (if available)
    positions = None
    feats_labels = None
    if feat_labels_available:
        feats_labels = bag_feats_df['label'].to_numpy()
        positions = list(bag_feats_df['position'])

    label = label.astype('float32')
    feats = feats.astype('float32')

    return label, feats, feats_labels, positions


def _load_data(bags_df, args):
    all_labels = []
    all_feats = []
    all_feats_labels = []
    all_positions = []
    all_slide_names = []

    feats_labels_available = True

    for i in tqdm(range(len(bags_df))):
        label, feats, feats_labels, positions = get_bag_feats(bags_df.iloc[i], args)
        all_labels.append(label)
        all_feats.append(feats)

        if feats_labels is None:
            feats_labels_available = False
        if feats_labels_available:
            all_feats_labels.append(feats_labels)
            all_positions.append(positions)
        all_slide_names.append(bags_df.iloc[i]['0'].split('/')[-1].split('.')[0])

    if not feats_labels_available:
        all_feats_labels = None
        all_positions = None

    return all_labels, all_feats, all_feats_labels, all_positions, all_slide_names


def _load_data_mp_worker(args):
    i, row, args = args
    label, feats, feats_labels, positions = get_bag_feats(row, args)
    slide_name = row['0'].split('/')[-1].split('.')[0]
    return label, feats, feats_labels, positions, slide_name


def _load_data_mp(bags_df, args):
    with mp.Pool(processes=args.num_processes) as pool:
        all_labels, all_feats, all_feats_labels, all_positions, all_slide_names = zip(
            *pool.map(_load_data_mp_worker, [(i, bags_df.iloc[i], args) for i in range(len(bags_df))])
        )
    all_labels, all_feats, all_feats_labels, all_positions, all_slide_names = (
        list(all_labels), list(all_feats), list(all_feats_labels), list(all_positions), list(all_slide_names)
    )

    feats_labels_available = all_feats_labels[0] is not None
    if not feats_labels_available:
        all_feats_labels = None
        all_positions = None
    return all_labels, all_feats, all_feats_labels, all_positions, all_slide_names


def load_data(dataframe, args):
    if args.use_mp:
        return _load_data_mp(dataframe, args)
    else:
        return _load_data(dataframe, args)


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def multi_label_roc(labels, predictions, num_classes, for_feats=False):
    thresholds = []
    thresholds_optimal = []
    aucs = []

    if len(predictions.shape) == 1 and not for_feats:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        if for_feats:
            label = labels
            prediction = predictions
        else:
            label = labels[:, c]
            prediction = predictions[:, c]

        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)

    return aucs, thresholds, thresholds_optimal


def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label >= threshold_optimal] = 1
    this_class_label[this_class_label < threshold_optimal] = 0
    bag_predictions = this_class_label
    accuracy = 1 - np.count_nonzero(np.array(bag_labels).astype(int) - bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def visualize_attentions(args, slide_name, bag_prediction, attentions, positions):
    color_dict = {1: [1, 0, 0], 0: [0, 1, 0]}
    color = color_dict.get(bag_prediction)

    color_map = np.zeros((np.amax(positions, 0)[0] + 1, np.amax(positions, 0)[1] + 1, 3))
    attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
    for k, pos in enumerate(positions):
        tile_color = np.asarray(color) * attentions[k]
        color_map[pos[0], pos[1]] = tile_color

    color_map = transform.resize(color_map, (color_map.shape[0] * 32, color_map.shape[1] * 32), order=0)
    attention_maps_path = os.path.join(f'tmp', 'output', args.wandb_run)
    os.makedirs(attention_maps_path, exist_ok=True)
    io.imsave(f'{attention_maps_path}/{slide_name}.png', img_as_ubyte(color_map))


def replace_key_names(d: dict, old_term: str, new_term: str):
    """
    Returns a new dict, where all the occurrences of the `old_term` is replaced with `new_term`
    """
    new_d = {}
    for old_key, value in d.items():
        new_key = old_key.replace(old_term, new_term)
        new_d[new_key] = value
    return new_d


def delete_files_in_folder(path):
    files = glob.glob(os.path.join(path, '*'))
    for f in files:
        os.remove(f)


def delete_files_for_epoch(base_dir, epoch):
    try:
        os.remove(os.path.join(base_dir, f'{epoch}.pth'))
    except FileNotFoundError as e:
        pass
    try:
        os.remove(os.path.join(base_dir, f'lambda_parameter_{epoch}'))
    except FileNotFoundError as e:
        pass
    try:
        os.remove(os.path.join(base_dir, f'thresholds_{epoch}.txt'))
    except FileNotFoundError as e:
        pass


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def check_layers(model_state_dict, weights_state_dict, header='', align=True):
    matched_layers, discarded_layers = 0, 0

    for name, param in model_state_dict.items():
        if name in weights_state_dict and param.size() == weights_state_dict[name].size():
            matched_layers += 1
        else:
            discarded_layers += 1

    for name, param in weights_state_dict.items():
        if name not in model_state_dict:
            discarded_layers += 1

    print(
        f'{header} >'
        f' Model: {len(model_state_dict.keys())} |'
        f' Weights: {len(weights_state_dict)} |'
        f' Matched: {matched_layers} |'
        f' Discarded: {discarded_layers}'
    )
    if align:
        layers_comparison_table = get_aligned_layers_comparison_table(model_state_dict, weights_state_dict, header)
    else:
        layers_comparison_table = get_layers_comparison_table(model_state_dict, weights_state_dict, header)
    print(layers_comparison_table)
    print('\n')


def get_aligned_layers_comparison_table(model_state_dict, weights_state_dict, header=''):
    model_layers = sorted(model_state_dict.keys())
    weight_layers = sorted(weights_state_dict.keys())

    matched_layers = []
    m_ptr = 0
    w_ptr = 0
    for i in range(max(len(model_layers), len(weight_layers))):
        try:
            cur_model_layer = model_layers[m_ptr]
        except IndexError:
            cur_model_layer = ''
        try:
            cur_weight_layer = weight_layers[w_ptr]
        except IndexError:
            cur_weight_layer = ''

        if '' in [cur_model_layer, cur_weight_layer]:
            matched_layers.append((cur_model_layer, cur_weight_layer))
            continue

        if cur_model_layer == cur_weight_layer:
            matched_layers.append((cur_model_layer, cur_weight_layer))
            m_ptr += 1
            w_ptr += 1
        elif cur_model_layer > cur_weight_layer:
            matched_layers.append(('', cur_weight_layer))
            w_ptr += 1
        elif cur_model_layer < cur_weight_layer:
            matched_layers.append((cur_model_layer, ''))
            m_ptr += 1

    table = tabulate(matched_layers, headers=[f"{header} Model", f"{header} Weights"], tablefmt="simple")
    return table


def get_layers_comparison_table(model_state_dict, weights_state_dict, header=''):
    model_layers = sorted(model_state_dict.keys())
    weight_layers = sorted(weights_state_dict.keys())

    matched_layers = list(itertools.zip_longest(model_layers, weight_layers, fillvalue=''))
    table = tabulate(matched_layers, headers=[f"{header} Model", f"{header} Weights"], tablefmt="simple")
    return table


def convert_dsmil_mil_dataset_format_to_our_format(bag_ins_list, args) -> Tuple[
    List[np.ndarray], List[np.ndarray], None, None]:
    """
        bag_ins_list [
                [bag_0_label (int), np.array([np.array(instance_0_embeddings), np.array(instance_0_embeddings), ...])]
                [bag_1_label (int), np.array([np.array(instance_0_embeddings), np.array(instance_0_embeddings), ...])]
                ...
            ]

        output ([bag_0_label, bag_1_label, bag_2_label, ...],
                [np.array(bag_0_embeddings), np.array(bag_1_embeddings), ...]
            )
    """
    all_labels = []
    all_feats = []
    for bag_label, bag_vector in bag_ins_list:
        bag_label_converted = np.expand_dims(
            np.array(int(np.clip(bag_label, 0, 1)), dtype=float), axis=0
        )
        bag_vector_converted = np.stack(bag_vector)[:, 0:args.feats_size]

        all_labels.append(bag_label_converted)
        all_feats.append(bag_vector_converted)

    return all_labels, all_feats, None, None


def cross_validation_set(bag_ins_list, num_folds: int, current_fold: int, valid_ratio: float):
    """
        copied from datasets/mil_dataset/mil_cross_validation
    """
    csv_list = deepcopy(bag_ins_list)
    n = int(len(csv_list) / num_folds)

    chunked = [csv_list[i:i + n] for i in range(0, len(csv_list), n)]

    test_list = chunked.pop(current_fold)
    train_valid_list = list(itertools.chain.from_iterable(chunked))  # this should be after the popping!

    train_list = train_valid_list[0:int(len(train_valid_list) * (1 - valid_ratio))]
    valid_list = train_valid_list[int(len(train_valid_list) * (1 - valid_ratio)):]
    return train_list, valid_list, test_list


def load_mil_data(args, mil_datasets_base_path='./datasets/mil_dataset'):
    dataset_file_name_mapping = {
        'musk1': 'musk1norm',
        'musk2': 'musk2norm',
        'elephant': 'data_100x100',
        'fox': 'data_100x100',
        'tiger': 'data_100x100',
    }
    dataset_folder_name_mapping = {
        'musk1': 'Musk',
        'musk2': 'Musk',
        'elephant': 'Elephant',
        'fox': 'Fox',
        'tiger': 'Tiger',
    }
    dataset_file_name = dataset_file_name_mapping[args.dataset]
    dataset_folder_name = dataset_folder_name_mapping[args.dataset]
    bag_ins_list_file_name = f'{dataset_file_name}_{args.cv_num_folds}folds_{args.cv_valid_ratio}split.pkl'
    with open(os.path.join(mil_datasets_base_path, dataset_folder_name, bag_ins_list_file_name), 'rb') as f:
        bag_ins_list = pickle.load(f)

    train_ins_list, valid_ins_list, test_ins_list = cross_validation_set(
        bag_ins_list, args.cv_num_folds, args.cv_current_fold, args.cv_valid_ratio
    )
    train_data = convert_dsmil_mil_dataset_format_to_our_format(train_ins_list, args)
    valid_data = convert_dsmil_mil_dataset_format_to_our_format(valid_ins_list, args)
    test_data = convert_dsmil_mil_dataset_format_to_our_format(test_ins_list, args)
    return train_data, valid_data, test_data


def compute_pos_weight(labels):
    """
    A utility function to use a weighted version of BinaryCrossEntropy loss
    for unbalanced datasets (Musk1, Musk2, Elephant, Tiger, Fox).
    """
    pos_count = 0
    for label in labels:
        pos_count = pos_count + np.clip(label, 0, 1)
    return (len(labels) - pos_count) / pos_count


## add essential functions for compute_feats (DINO version)
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
