"""
Author: Vilem Gottwald

Module containing utility functions.
"""

import os
from pathlib import Path
import json
import pickle

# Path to project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.relative_to(Path.cwd())

# Path to data directory
DATA_PATH = PROJECT_ROOT / "data"


# json load
with open(DATA_PATH / "dataset" / "class_ids.json") as json_file:
    CLASS2ID = json.load(json_file)

with open(DATA_PATH / "dataset" / "dataset_columns.json") as json_file:
    COL_IDX = json.load(json_file)

with open(DATA_PATH / "parsing" / "frame_times.pkl", "rb") as f:
    FRAME_TIMES = pickle.load(f)


def listdir_paths(directory):
    """
    Returns list of paths to files in given directory.

    :param directory: Path to directory

    :return: List of paths to files in given directory
    """
    return sorted([os.path.join(directory, file) for file in os.listdir(directory)])


def replace_dir(filepath, new_dir):
    """
    Returns filepath with new_dir as its directory.

    :param filepath: Path to file
    :param new_dir: Path to new directory

    :return: Path to file with new_dir as its directory
    """
    return os.path.join(new_dir, os.path.basename(filepath))
