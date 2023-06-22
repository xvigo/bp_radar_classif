"""
Author: Vilem Gottwald

Module containing common paths to project data and other constants.
"""

import os
from pathlib import Path
import json
import pickle
import numpy as np
import joblib

# Path to project root directory
try:
    PROJECT_ROOT = Path(globals()["_dh"][0]).resolve().parent.parent
except KeyError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to data directory
DATA_PATH = PROJECT_ROOT / "data"
SCALER_PATH = str(DATA_PATH / "training" / "normalization_scaler.save")
DATASET_SPLIT_IDX = 28162

# Dataset path
DATASET_PATH = str(DATA_PATH / "dataset" / "dataset_gt")

# Extracted features and gt classes paths
FEATURES_PATH = str(DATA_PATH / "training" / "features" / "features.npy")
CLASSES_PATH = str(DATA_PATH / "training" / "features" / "gt_classes.npy")


DETECTIONS_PATH = str(DATA_PATH / "detection" / "detections_predicted")
DETECTIONS_GT_CLASSES_PATH = str(
    DATA_PATH / "detection" / "ious_and_gt_classes" / "gt_classes.npy"
)
DETECTIONS_IOU_PATH = str(DATA_PATH / "detection" / "ious_and_gt_classes" / "ious.npy")
DETECTED_FEATURES_PATH = str(DATA_PATH / "detection" / "extracted" / "features.npy")
DETECTED_OBJECTS_IDS_PATH = str(
    DATA_PATH / "detection" / "extracted" / "object_ids.npy"
)
FIRST_TEST_FRAME_IDX = 6211
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# json load
with open(DATA_PATH / "dataset" / "class_ids.json") as json_file:
    CLASS2ID = json.load(json_file)

with open(DATA_PATH / "dataset" / "dataset_columns.json") as json_file:
    COL_IDX = json.load(json_file)

with open(DATA_PATH / "parsing" / "frame_times.pkl", "rb") as f:
    FRAME_TIMES = pickle.load(f)


def split_data(data, offset, test_ratio=0.2):
    """Split data into training and validation sets with given offset

    :param data: data to split
    :param offset: offset of the first sample in the data
    :param test_ratio: ratio of test data

    :return: tuple of training and validation data
    """
    test_samples_cnt = int(data.shape[0] * test_ratio)

    # sequence starting ar offset us used as test data
    test_data = data[offset : offset + test_samples_cnt]

    # train data - concatenate data before and after test data
    train_data = np.concatenate(
        (data[:offset], data[offset + test_samples_cnt :]), axis=0
    )
    return train_data, test_data


def normalize_features(data, scaler_path=SCALER_PATH):
    """Normalize data using scaler

    :param scaler_path: path to scaler
    :param data: data to normalize

    :return: normalized data
    """
    scaler = joblib.load(scaler_path)
    return scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
