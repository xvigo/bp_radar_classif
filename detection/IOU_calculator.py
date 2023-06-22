"""
Author: Vilem Gottwald

Module for calculating the IOU of predicted objects with ground truth objects.
"""

import os
import numpy as np
from utils import *


def print_sequences(frames_set):
    """
    Prints sequences of frames in given set of frames.

    :param frames_set: Set of frames

    :return: None
    """
    frames = sorted(list(frames_set))
    sequence_start = frames[0]
    prev_frame = frames[0]

    for frame in frames[1:]:
        if prev_frame + 1 != frame:
            print(
                f"Sequence: {sequence_start} - {prev_frame}, length: {prev_frame - sequence_start + 1}"
            )
            sequence_start = frame

        prev_frame = frame

    # If last frame was a standalone sequence
    if sequence_start:
        print(
            f"Sequence: {sequence_start} - {prev_frame}, length: {prev_frame - sequence_start + 1}"
        )


def generate_IOU(
    ground_truth_dir,
    prediction_dir,
    output_dir,
    start_idx=0,
    end_idx=None,
    print_non_perfect_frames=False,
):
    """
    Calculates the IOU of each predicted object with the ground truth objects.
    Saves the mean IOU of each predicted object as well as the class of the ground truth object with the highest IOU.
    Also saves the frames where the IOU is not perfect (IOU != 1.0) as well as the frames where multiple ground truth objects overlap with a single predicted object.

    :param ground_truth_dir: Path to directory containing ground truth .ply files
    :param prediction_dir: Path to directory containing predicted .ply files
    :param output_dir: Path to directory where results should be saved
    :param start_idx: Index of first frame to be processed
    :param end_idx: Index of last frame to be processed

    :return: None
    """

    # Get filepaths to all ply files in given directory
    prediction_filepaths = listdir_paths(prediction_dir)

    not_perfect_frames = set() 
    different_obj_cnt_frames = set()
    counter = 0

    # Set stop index for progress printing
    if end_idx is None:
        end_idx = len(prediction_filepaths)

    all_objects_ious = []
    all_object_classes = []

    # For each frame in dataset
    for frame_idx, pred_filepath in enumerate(
        prediction_filepaths[start_idx:end_idx], start_idx
    ):
        np_prediction = np.load(pred_filepath)
        np_ground_truth = np.load(replace_dir(pred_filepath, ground_truth_dir))

        # Get all unique cluster labels
        pred_objects = np.unique(np_prediction[:, COL_IDX["object_id"]]).tolist()
        for pred_object in pred_objects:
            pred_object_mask = np_prediction[:, COL_IDX["object_id"]] == pred_object
            object_points_ids = np_prediction[pred_object_mask, COL_IDX["point_id"]]

            # Skip objects with less than 4 points
            if object_points_ids.shape[0] < 4:
                continue

            object_ious = []
            object_classes = []

            # Get all gt objects that contain any of the points in the current object
            gt_overlapping_objects = np.unique(
                np_ground_truth[
                    np.isin(np_ground_truth[:, COL_IDX["point_id"]], object_points_ids),
                    COL_IDX["object_id"],
                ]
            ).tolist()

            # Calculate IOU for each overlapping gt object and get class of gt object
            for gt_object in gt_overlapping_objects:
                gt_object_mask = np_ground_truth[:, COL_IDX["object_id"]] == gt_object
                gt_points_ids = np_ground_truth[gt_object_mask, COL_IDX["point_id"]]

                # Skip objects with less than 4 points
                if object_points_ids.shape[0] < 4:
                    continue

                # Calculate IOU of current object and gt object
                intersection = np.intersect1d(object_points_ids, gt_points_ids)
                union = np.union1d(object_points_ids, gt_points_ids)
                iou = intersection.shape[0] / union.shape[0]
                object_ious.append(iou)

                # Get class of gt object
                gt_class = np_ground_truth[gt_object_mask, COL_IDX["class_id"]][0]
                object_classes.append(gt_class)

            # Add frame to set if more than one gt object overlaps with current object
            if len(object_classes) > 1:
                different_obj_cnt_frames.add(frame_idx)
                counter += 1

            # Calculate mean IOU of current object
            iou_mean = np.mean(object_ious)
            all_objects_ious.append(iou_mean)

            if iou_mean != 1.0:
                not_perfect_frames.add(frame_idx)

            # Select class of gt object with highest IOU
            gt_class = object_classes[np.argmax(object_ious)]
            all_object_classes.append(gt_class)

    # Print mean IOU and other statistics
    iou_len = len(all_objects_ious)
    iou_mean = np.mean(all_objects_ious)
    print(f"Mean IOU: {iou_mean} calculated from {iou_len} predicted objects!")

    if print_non_perfect_frames:
        print(f"Found {counter} frames with multiple GT overlapping single predicted:")
        print_sequences(different_obj_cnt_frames)

    # Save all frames IOUs and classes as .npy
    np.save(os.path.join(output_dir, "ious.npy"), np.array(all_objects_ious))
    np.save(os.path.join(output_dir, "gt_classes.npy"), np.array(all_object_classes))

    # Save all frames with non-perfect IOU as .npy
    np.save(
        os.path.join(output_dir, "not_perfect_frames.npy"),
        np.array(sorted(not_perfect_frames)),
    )

    print("number of not perfect frames: ", len(not_perfect_frames))


if __name__ == "__main__":

    gt_dir = str(DATA_PATH / "dataset" / "dataset_gt")
    pred_dir = str(DATA_PATH / "detection" / "detections_predicted")
    output_dir = str(DATA_PATH / "detection")
    # make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    generate_IOU(gt_dir, pred_dir, output_dir, start_idx=0, end_idx=10501)
