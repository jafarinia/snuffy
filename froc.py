# CAMELYON16 data set by Computational Pathology Group of Radboud University
# Medical Centre
#
# CAMELYON16 data set is available under CC0.
#
# This file has been modified to include additional functionality: Copyright (c) 2024 Hossein Jafarinia
"""
Evaluation code for the CAMELYON16 challenge on cancer metastases detection.
"""

import sys

sys.path.append('/opt/ASAP/bin')
import multiresolutionimageinterface as mir
import pandas as ps
import numpy as np
import scipy.ndimage
import skimage.measure
import matplotlib.pyplot as plt
import bisect
import argparse
import os
import multiprocessing as mp


# ----------------------------------------------------------------------------------------------------

def load_detections(detection_path, mask_path, level):
    """
    Load detection (probability, X coordinate, Y coordinate) tuples from detection CSV file.

    After loading the coordinates the pixel indices are adjusted to the level of the whole-slide image that corresponds to the evaluation
    level in the mask image.

    Args:
        detection_path (str): Path of detection CSV file.
        mask_path (str): Path of the mask image.
        level (int): Evaluation level of the mask image.

    Returns:
        list: List of (probability, row, col) detection tuples.
    """

    # Load (probability, y coordinate, x coordinate) tuple list from the detection CSV file. Note that it is read in numpy (row, col) order.
    #
    detection_table = ps.read_csv(detection_path)
    detection_items = [(detection_row['p'], int(detection_row['y']), int(detection_row['x'])) for _, detection_row in
                       detection_table.iterrows()]

    # Load the downsampling factor at the evaluated level of the mask.
    #
    mask_image = mir.MultiResolutionImageReader().open(mask_path)
    level_downsampling = mask_image.getLevelDownsample(level=level)
    # mask_image.close()

    # Rescale the prediction coordinates from level 0 of the WSI to the level that matches the mask at the evaluation level.
    #
    detection_items = [
        (detection[0], round(detection[1] / level_downsampling), round(detection[2] / level_downsampling)) for detection
        in detection_items]

    return detection_items


# ----------------------------------------------------------------------------------------------------

def compute_evaluation_mask(mask_path, level, include_itcs):
    """
    Computes the evaluation mask and the list of region labels that contains Isolated Tumor Cells (ITCs).

    The mask is a label image, that is calculated by the following steps:
        1. Load the content of the mask at the given level.
        2. Select the tumor regions.
        3. Dilating the regions with a distance of a few tumor cells.
        4. Label the disjunctive regions.

    Args:
        mask_path (str): The mask TIF file to load.
        level (int): Processing level.
        include_itcs (bool): Flag to control skipping the ITCs in evaluation.

    Returns:
        (np.ndarray, set): Evaluation mask, and set of labels containing ITCs.
    """

    # Hardwired constants.
    #
    tumor_label_value = 2  # The label value of tumor in the TIF masks is 2.
    dilation_distance = 75.0  # 75 um is approximately 5 tumor cells

    # Read the content of the image to memory at the given level.
    #
    mask_image = mir.MultiResolutionImageReader().open(mask_path)
    level_width, level_height = mask_image.getLevelDimensions(level=level)
    image_array = mask_image.getUCharPatch(startX=0, startY=0, width=level_width, height=level_height, level=level)
    image_array = image_array.squeeze()

    # Calculate the distance from the edges or the tumor regions.
    #
    image_negative_array = np.not_equal(image_array, tumor_label_value)
    image_distance_array = scipy.ndimage.distance_transform_edt(input=image_negative_array)

    # Add a few cells of distance around the tumor.
    #
    image_spacing = mask_image.getSpacing()[0]
    image_downsampling = mask_image.getLevelDownsample(level=level)
    image_level_spacing = image_spacing * image_downsampling
    distance_threshold_pixels = dilation_distance / (image_level_spacing * 2.0)
    image_binary_array = np.less(image_distance_array, distance_threshold_pixels)

    # Fill the holes in the dilated tumor mask and label the regions.
    #
    # image_filled_array = scipy.ndimage.morphology.binary_fill_holes(input=image_binary_array)
    image_filled_array = scipy.ndimage.binary_fill_holes(input=image_binary_array)
    # image_evaluation_mask = skimage.measure.label(input=image_filled_array, connectivity=2)
    image_evaluation_mask = skimage.measure.label(label_image=image_filled_array, connectivity=2)

    # Collect the list of region labels that are ITC. A region is considered ITC if its longest diameter is below 200 um.
    #
    if include_itcs:
        itc_labels = set()
    else:
        itc_size_threshold = (200.0 + dilation_distance) / image_level_spacing
        region_properties = skimage.measure.regionprops(label_image=image_evaluation_mask)
        itc_labels = set(label_index + 1 for label_index in range(len(region_properties)) if
                         region_properties[label_index].major_axis_length < itc_size_threshold)

    return image_evaluation_mask, itc_labels


# ----------------------------------------------------------------------------------------------------

def compute_probabilities(detection_items, evaluation_mask, itc_labels):
    """
    Generate true positive and false positive stats for the analyzed image.

    Args:
        detection_items (list): List of (probability, row coordinate, column coordinate) detection items.
        evaluation_mask (np.ndarray, None): Evaluation mask.
        itc_labels (set): Set of ITC labels.

    Returns:
        (list, list, int): List of false positive, and list of true positive detection probabilities, and the number of non ITC tumors.
    """

    # Check if the slide contains tumor: there is a mask available.
    #
    if evaluation_mask is not None:
        # Initialize result.
        #
        max_label = evaluation_mask.max()
        fp_probs = []
        tp_probs = [0.0] * (max_label + 1)

        # Check if the detection hit a a tumor area or normal tissue, but discard the ITC results.
        #
        for detection in detection_items:
            hit_label = evaluation_mask[detection[1:]]

            if hit_label == 0:
                fp_probs.append(detection[0])

            elif hit_label not in itc_labels:
                if tp_probs[hit_label] < detection[0]:
                    tp_probs[hit_label] = detection[0]

        # Calculate the number of tumor areas.
        #
        number_of_tumors = max_label - len(itc_labels)

    else:
        # Initialize result.
        #
        fp_probs = []
        tp_probs = [0.0]

        # This slide does contain tumor, all detections are false positive.
        #
        for detection in detection_items:
            fp_probs.append(detection[0])

        # The number of tumors is 0.
        #
        number_of_tumors = 0

    # Drop the first, unused probability.
    #
    tp_probs = tp_probs[1:]

    return fp_probs, tp_probs, number_of_tumors


# ----------------------------------------------------------------------------------------------------

def mp_greater_equal(args):
    aggregated_fps, aggregated_tps, threshold = args
    if threshold == -1:
        return 0, 0
    return np.greater_equal(aggregated_fps, threshold).sum(), np.greater_equal(aggregated_tps, threshold).sum()


def mp_compute_froc(froc_data, num_processes):
    """
    Generates the data required for plotting the FROC curve.

    Args:
        froc_data (dict):      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        (list, list):  A list of the average number of false positives per image for different thresholds,
            and a list of overall sensitivity of the system for different thresholds.
    """

    # Aggregate the results over all the images.
    #
    aggregated_fps = [prob for froc_item in froc_data.values() for prob in froc_item['fp']]
    aggregated_tps = [prob for froc_item in froc_data.values() for prob in froc_item['tp']]
    all_probs = sorted(set(aggregated_fps + aggregated_tps) - {0.0})
    image_count = len(froc_data)
    total_tumor_count = sum(froc_item['count'] for froc_item in froc_data.values())

    # Count item counts with increasing thresholds.
    #
    aggregated_fps = np.asarray(a=aggregated_fps, dtype=np.float64)
    aggregated_tps = np.asarray(a=aggregated_tps, dtype=np.float64)

    with mp.Pool(num_processes) as pool:
        total_fps, total_tps = zip(*pool.map(mp_greater_equal,
                                             [(aggregated_fps, aggregated_tps, threshold) for threshold in
                                              all_probs + [-1]]))

    # Finalize the values.
    #
    total_fps = [count / image_count for count in total_fps]
    total_sensitivity = [count / total_tumor_count for count in total_tps]

    return total_fps, total_sensitivity, all_probs


def compute_froc(froc_data):
    """
    Generates the data required for plotting the FROC curve.

    Args:
        froc_data (dict):      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        (list, list):  A list of the average number of false positives per image for different thresholds,
            and a list of overall sensitivity of the system for different thresholds.
    """

    # Aggregate the results over all the images.
    #
    aggregated_fps = [prob for froc_item in froc_data.values() for prob in froc_item['fp']]
    aggregated_tps = [prob for froc_item in froc_data.values() for prob in froc_item['tp']]
    all_probs = sorted(set(aggregated_fps + aggregated_tps) - {0.0})
    image_count = len(froc_data)
    total_tumor_count = sum(froc_item['count'] for froc_item in froc_data.values())

    # Count item counts with increasing thresholds.
    #
    aggregated_fps = np.asarray(a=aggregated_fps, dtype=np.float64)
    aggregated_tps = np.asarray(a=aggregated_tps, dtype=np.float64)

    total_fps = []
    total_tps = []
    for threshold in all_probs:
        total_fps.append(np.greater_equal(aggregated_fps, threshold).sum())
        total_tps.append(np.greater_equal(aggregated_tps, threshold).sum())

    total_fps.append(0)
    total_tps.append(0)

    # Finalize the values.
    #
    total_fps = [count / image_count for count in total_fps]
    total_sensitivity = [count / total_tumor_count for count in total_tps]

    return total_fps, total_sensitivity, all_probs


# ----------------------------------------------------------------------------------------------------

def compute_score(average_fps, sensitivities):  # probably should combine this with area_under_curve(x, y)
    """
    Compute the second evaluation metric of the challenge: average sensitivity at 6 predefined false
    positive rates: 1/4, 1/2, 1, 2, 4, and 8 FPs per whole slide image.

    Args:
        average_fps (list): List of the average number of false positives per image for different thresholds.
        sensitivities (list): List of overall sensitivity of the system for different thresholds.

    Returns:
        float: Computed score.
    """

    average_fps_r = list(reversed(average_fps))
    sensitivities_r = list(reversed(sensitivities))

    threshold_count = len(sensitivities_r)
    target_fp_items = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    target_sum = sum(
        sensitivities_r[min(bisect.bisect_left(average_fps_r, target_fp), threshold_count - 1)] for target_fp in
        target_fp_items)

    return target_sum / len(target_fp_items)


# ----------------------------------------------------------------------------------------------------

def save_results(result_file_path, average_fps, sensitivities):
    """
    Save the results.

    Args:
        result_file_path (str): Result file path.
        average_fps (list): List of the average number of false positives per image for different thresholds.
        sensitivities (list): List of overall sensitivity of the system for different thresholds.
    """

    result_df = ps.DataFrame.from_dict(data={'Average FP Counts': average_fps, 'Overall Sensitivities': sensitivities},
                                       dtype=np.float64)
    result_df.to_csv(path_or_buf=result_file_path, columns=['Average FP Counts', 'Overall Sensitivities'], index=False)


# ----------------------------------------------------------------------------------------------------

def plot_froc(average_fps, sensitivities, path, plot_prefix):
    """
    Plot the FROC curve.

    Args:
        average_fps (list): List of the average number of false positives per image for different thresholds.
        sensitivities (list): List of overall sensitivity of the system for different thresholds.
    """

    fig = plt.figure()
    plt.xlabel('Average Number of False Positives')
    plt.ylabel('Metastasis Detection Sensitivity')
    plt.title(f'FROC Curve for {plot_prefix}')
    plt.plot(average_fps, sensitivities, linestyle='-', color='black')
    plt.savefig(os.path.join(path, f'froc_{plot_prefix}.png'))
    plt.show()
    plt.close()


# ----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, str, str, bool, bool): The parsed command line arguments: reference CSV file path, mask folder path with the mask TIF files,
            detection folder path with the detection CSV files, result CSV table path, include ITCs flag, and plot curve flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Compute FROC on the CAMELYON16 test set.')

    argument_parser.add_argument('-r', '--reference', required=True, type=str, default='./reference.csv',
                                 help='reference CSV file path')
    argument_parser.add_argument('-m', '--masks', required=True, type=str, default='../masks',
                                 help='reference mask folder path')
    argument_parser.add_argument('-d', '--detections', required=True, type=str, help='detection file folder path')
    argument_parser.add_argument('-o', '--result', required=False, type=str, default=None,
                                 help='result table file path')
    argument_parser.add_argument('-t', '--itc', action='store_true', help='include ITCs in calculation')
    argument_parser.add_argument('-p', '--plot', action='store_true', help='plot curve')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())
    parsed_reference_path = arguments['reference']
    parsed_masks_path = arguments['masks']
    parsed_detections_path = arguments['detections']
    parsed_result_path = arguments['result']
    parsed_itc_flag = arguments['itc']
    parsed_plot_flag = arguments['plot']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Reference path: {path}'.format(path=parsed_reference_path))
    print('Masks path: {path}'.format(path=parsed_masks_path))
    print('Detections path: {path}'.format(path=parsed_detections_path))
    print('Result path: {path}'.format(path=parsed_result_path))
    print('Include ITCs: {flag}'.format(flag=parsed_itc_flag))
    print('Plot curve: {flag}'.format(flag=parsed_plot_flag))
    print('')

    return parsed_reference_path, parsed_masks_path, parsed_detections_path, parsed_result_path, parsed_itc_flag, parsed_plot_flag


# ----------------------------------------------------------------------------------------------------

def load_detections_list(detections, mask_path, level):
    """
    Load detection (probability, X coordinate, Y coordinate) tuples from detection CSV file.

    After loading the coordinates the pixel indices are adjusted to the level of the whole-slide image that corresponds to the evaluation
    level in the mask image.

    Args:
        detection_path (str): Path of detection CSV file.
        mask_path (str): Path of the mask image.
        level (int): Evaluation level of the mask image.

    Returns:
        list: List of (probability, row, col) detection tuples.
    """

    # Load (probability, y coordinate, x coordinate) tuple list from the detection CSV file. Note that it is read in numpy (row, col) order.
    #
    detection_items = [(detection[0], detection[2], detection[1]) for detection in detections]

    # Load the downsampling factor at the evaluated level of the mask.
    #
    mask_image = mir.MultiResolutionImageReader().open(mask_path)
    level_downsampling = mask_image.getLevelDownsample(level=level)
    # mask_image.close()

    # Rescale the prediction coordinates from level 0 of the WSI to the level that matches the mask at the evaluation level.
    #
    detection_items = [
        (detection[0], round(detection[1] / level_downsampling), round(detection[2] / level_downsampling)) for detection
        in detection_items]

    return detection_items


def mp_froc_data_list(args):
    image_name, masks_folder_path, detections, evaluation_mask_level, include_itcs, slide_type = args
    mask_path = os.path.join(masks_folder_path, '{image}_mask.tif'.format(image=image_name))
    print(f'Processing: {image_name}')
    detection_items = load_detections_list(detections[image_name], mask_path, evaluation_mask_level)
    evaluation_mask, itc_labels = compute_evaluation_mask(mask_path=mask_path, level=evaluation_mask_level,
                                                          include_itcs=include_itcs) if slide_type == 'tumor' else (
        None, set())
    fp_probs, tp_probs, number_of_tumors = compute_probabilities(detection_items=detection_items,
                                                                 evaluation_mask=evaluation_mask, itc_labels=itc_labels)
    return fp_probs, tp_probs, number_of_tumors


def mp_computeFROC_list_no_cache(reference_file_path, masks_folder_path, detections, result_file_path,
                                 include_itcs, plot_curve, evaluation_mask_level, images_to_calculate_for, plot_path,
                                 plot_prefix, num_processes):
    global use_cache
    use_cache = False
    """Entry point. Calculates and plots the FROC curve."""

    # Hardwired constants.
    #
    # evaluation_mask_level = 5

    # Collect the command line arguments.
    #
    # reference_file_path, masks_folder_path, detections_folder_path, result_file_path, include_itcs, plot_curve = collect_arguments()

    # ITCs were not included in the CAMELYON16 challenge.
    #
    if include_itcs:
        print('Warning: ITCs are included in the FROC calculation. The CAMELYON16 challenge did not include ITCs.')
        print('')

    # Process each test image.
    #
    froc_data = dict()

    reference_table = ps.read_csv(reference_file_path)
    mp_list = []
    for _, reference_row in reference_table.iterrows():
        image_name, _ = os.path.splitext(reference_row['image'])
        if image_name.split('.')[0] not in images_to_calculate_for:
            continue
        mp_list.append(
            (image_name, masks_folder_path, detections, evaluation_mask_level, include_itcs, reference_row['type']))
    with mp.Pool(num_processes) as pool:
        fp_probs, tp_probs, number_of_tumors = zip(*pool.map(mp_froc_data_list, mp_list))
    idx = 0
    for _, reference_row in reference_table.iterrows():
        image_name, _ = os.path.splitext(reference_row['image'])
        if image_name.split('.')[0] not in images_to_calculate_for:
            continue
        froc_data[reference_row['image']] = {'fp': fp_probs[idx], 'tp': tp_probs[idx], 'count': number_of_tumors[idx]}
        idx += 1
    # Compute FROC.
    #
    average_fps, sensitivities, thresholds = mp_compute_froc(froc_data=froc_data, num_processes=num_processes)

    # Compute score.
    #
    challenge_score = compute_score(average_fps=average_fps, sensitivities=sensitivities)

    print('')
    print('Score: {score}'.format(score=challenge_score))

    # Save the results.
    #
    if result_file_path:
        save_results(result_file_path=result_file_path, average_fps=average_fps, sensitivities=sensitivities)

    # Plot FROC curve.
    #
    if plot_curve:
        plot_froc(average_fps=average_fps, sensitivities=sensitivities, path=plot_path,
                  plot_prefix=plot_prefix)  # make it save if it doesn't

    return challenge_score
