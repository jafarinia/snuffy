import argparse
import itertools
import os
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def get_data(file_path):
    """
    Reads an SVM file containing MIL dataset entries and extracts relevant information such as
    instance IDs, bag IDs, class IDs, and feature vectors for each row.
    The expected format for each row in the CSV file is:
     "ID:BagID:ClassID Feature1:Value1 Feature2:Value2 ...".

    Parameters:
    - file_path (str): The path to the CSV file containing the MIL dataset entries.

    Returns:
    - list: A list of lists, where each inner list contains the instance ID, bag ID, class ID, and
            feature vector as a NumPy array for each row in the CSV file. This structured data is
            suitable for training MIL models.
    """
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = df[df.columns[0]]
    data_list = []
    for i in range(0, df.shape[0]):
        data = str(df.iloc[i]).split(' ')
        ids = data[0].split(':')
        instance_id = int(ids[0])
        bag_id = int(ids[1])
        class_id = int(ids[2])
        data = data[1:]
        feature_vector = np.zeros(len(data))
        for i, feature in enumerate(data):
            feature_data = feature.split(':')
            if len(feature_data) == 2:
                feature_vector[i] = feature_data[1]
        data_list.append([instance_id, bag_id, class_id, feature_vector])
    return data_list


def get_bag(data, idb):
    data_array = np.array(data, dtype=object)
    bag_id = data_array[:, 1]
    return data_array[np.where(bag_id == idb)]


def find_admissible_shuffle(args, bag_ins_list):
    """
    Iterates to find a shuffle ensuring each cross-validation fold has both positive and negative bags in
     training, validation, and testing sets for MIL datasets.

    Parameters:
    - args: An object containing arguments necessary for the operation, including:
      - num_folds (int): The number of folds for cross-validation.
      - train_valid_ratio (float): The ratio defining the split between training and validation sets.
    - bag_ins_list (list): A list of bags, where each bag is represented as a tuple containing:
      - The label of the bag (int): 0 for negative, 1 for positive.
      - A numpy array of instance embeddings (numpy.ndarray): Each element is a numpy array representing
        the embedding of an instance within the bag.

    Returns:
    - list: The shuffled list of bags that satisfies the condition of having both positive and negative bags
            in all splits (training, validation, testing) for each fold in the cross-validation setup.
    """
    found_valid_shuffle = False
    while not found_valid_shuffle:
        bag_ins_list = shuffle(bag_ins_list)
        for k in range(0, args.num_folds):
            train_ins_list, valid_ins_list, test_ins_list = cross_validation_set(
                bag_ins_list, num_folds=args.num_folds, current_fold=k, valid_ratio=args.train_valid_ratio
            )

            train_bags_labels = [np.clip(bag[0], 0, 1) for bag in train_ins_list]
            valid_bags_labels = [np.clip(bag[0], 0, 1) for bag in valid_ins_list]
            test_bags_labels = [np.clip(bag[0], 0, 1) for bag in test_ins_list]

            if not (0 in train_bags_labels and 1 in train_bags_labels):
                break
            if not (0 in valid_bags_labels and 1 in valid_bags_labels):
                break
            if not (0 in test_bags_labels and 1 in test_bags_labels):
                break
            found_valid_shuffle = True

    return bag_ins_list


def cross_validation_set(bag_ins_list, num_folds: int, current_fold: int, valid_ratio: float):
    """
    Splits a dataset into training, validation, and testing sets for k-fold cross-validation.
    The function divides the dataset into `num_folds` equal parts, assigns one part as the test set,
    and splits the remaining data into training and validation sets based on the `valid_ratio`.

    Parameters:
    - bag_ins_list (list): The original dataset containing bags of instances.
    - num_folds (int): The total number of folds for cross-validation.
    - current_fold (int): The index of the current fold used as the test set.
    - valid_ratio (float): The ratio of the validation set size relative to the combined training and validation set sizes.

    Returns:
    - tuple: A tuple containing three lists representing the training, validation, and testing datasets.
    """
    csv_list = deepcopy(bag_ins_list)
    n = int(len(csv_list) / num_folds)

    chunked = [csv_list[i:i + n] for i in range(0, len(csv_list), n)]

    test_list = chunked.pop(current_fold)
    train_valid_list = list(itertools.chain.from_iterable(chunked))  # this should be after the popping!

    train_list = train_valid_list[0:int(len(train_valid_list) * (1 - valid_ratio))]
    valid_list = train_valid_list[int(len(train_valid_list) * (1 - valid_ratio)):]
    return train_list, valid_list, test_list


def main(args, datasets_base_path='./'):
    """
    Loads a specified dataset, processes it into a format suitable for MIL,
     performs an admissible shuffle for cross-validation, and saves the processed dataset.

    Parameters:
    - args: Command-line arguments object containing settings for the experiment, including:
      - dataset (str): Identifier of the dataset to load from the registry.
      - num_folds (int): Number of folds for cross-validation.
      - train_valid_ratio (float): Ratio for splitting training and validation sets.
    - datasets_base_path (str, optional): Base path to the directory containing the datasets. Defaults to './'.

    Saves the processed and shuffled dataset to a file named according to the dataset, fold count, and split ratio.
    """
    mil_dataset_registry = {
        'musk1': ('Musk', 'musk1norm.svm', 166),
        'musk2': ('Musk', 'musk2norm.svm', 166),
        'elephant': ('Elephant', 'data_100x100.svm', 230),
        'fox': ('Fox', 'data_100x100.svm', 230),
        'tiger': ('Tiger', 'data_100x100.svm', 230),
    }
    dataset_folder, dataset_file, args.feats_size = mil_dataset_registry[args.dataset]
    data_all = get_data(os.path.join(datasets_base_path, dataset_folder, dataset_file))

    bag_ins_list = []
    num_bag = data_all[-1][1] + 1
    for i in range(num_bag):
        bag_data = get_bag(data_all, i)
        bag_label = bag_data[0, 2]
        bag_vector = bag_data[:, 3]
        bag_ins_list.append([bag_label, bag_vector])

    bag_ins_list = find_admissible_shuffle(args, bag_ins_list)
    file_name = f'{Path(dataset_file).stem}_{args.num_folds}folds_{args.train_valid_ratio}split.pkl'
    with open(os.path.join(datasets_base_path, dataset_folder, file_name), 'wb') as f:
        pickle.dump(bag_ins_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Dataset Cross-Validation')
    parser.add_argument('--dataset', default='musk1', type=str,
                        help='Choose MIL datasets from: musk1, musk2, elephant, fox, tiger [musk1]')
    parser.add_argument('--num_folds', default=10, type=int, help='Number of cross validation fold [10]')
    parser.add_argument('--train_valid_ratio', default=0.2, type=float, help='Train/Valid ratio')
    args = parser.parse_args()

    main(args)
