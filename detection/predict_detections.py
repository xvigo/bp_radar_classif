"""
Author: Vilem Gottwald

Module for generating detections from point clouds and saving them to .npy files.
"""

import os
import numpy as np
from pyntcloud import PyntCloud
from clusterer import Clusterer
from utils import *


def generate_detections(
    points_dirpath,
    dataset_dirpath,
    start_idx=0,
    end_idx=None,
):
    """
    Generates detections for each point cloud in given directory.

    :param points_dirpath: Path to directory containing point cloud .ply files
    :param dataset_dirpath: Path to directory where detections should be saved
    :param start_idx: Index of first frame to be processed
    :param end_idx: Index of last frame to be processed

    :return: None
    """

    # Get filepaths to all ply files in given directory

    ply_filepaths = listdir_paths(points_dirpath)

    # Set stop index for progress printing
    if end_idx is None:
        end_idx = len(ply_filepaths)

    # Process each point cloud file
    for i, ply_filepath in enumerate(ply_filepaths[start_idx:end_idx], start_idx):

        # Load point cloud from file
        try:
            cloud = PyntCloud.from_file(ply_filepath)
            points_df = cloud.points
        except Exception as e:
            print(ply_filepath, e)
            continue

        # Cluster points by lanes
        clusterer = Clusterer()
        clustered_points = clusterer.cluster(points_df)
        # 'x', 'y', 'z', 'snr', 'noise', 'velocity', 'y_orig', 'total_seconds', pointID', 'cluster'

        # fix datetime
        clustered_points[:, COL_IDX["total_seconds"]] = list(
            map(
                lambda x: FRAME_TIMES[x].timestamp(),
                clustered_points[:, COL_IDX["point_id"]] // 1000,
            )
        )

        # make object ids that are bigger than max_object_id fo
        # save numpy array of bboxes
        filename = os.path.splitext(os.path.basename(ply_filepath))[0] + ".npy"
        np.save(os.path.join(dataset_dirpath, filename), clustered_points)

        # Print progress
        print(f"\r {filename} created! {i}/{end_idx - 1}", end="")


PLY_DIR = str(DATA_PATH / "labeling" / "pointclouds")
OUT_PATH = str(DATA_PATH / "detection" / "detections_predicted")

# make sure the output directory exists
os.makedirs(OUT_PATH, exist_ok=True)

generate_detections(
    PLY_DIR,
    OUT_PATH,
    start_idx=0,
    end_idx=10501,
)
