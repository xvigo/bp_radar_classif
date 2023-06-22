"""
Author: Vilem Gottwald

Module for generating labelCloud labels in centroid format.
"""

import json
import os
import numpy as np


def create_json(
    filename: str, objects: list, ply_folder: str, labels_folder: str
) -> None:
    """
    Generates a labelCloud label in centroid format as JSON.

    :param filename: Name of the corresponding point cloud file.
    :param objects: List of objects in the frame.
    :param ply_folder: Path to the folder containing the ply files.
    :param labels_folder: Path to the folder where the JSON file will be saved.

    :return: None
    """
    # Fill data to the template
    data = {
        "folder": ply_folder,
        "filename": filename,
        "path": os.path.join(ply_folder, filename),
        "objects": objects,
    }

    # Construct the output file path for the JSON file
    out_filename = os.path.splitext(filename)[0]
    out_file_path = os.path.join(labels_folder, out_filename + ".json")

    # Save the JSON file
    with open(out_file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


def get_bounding_box(points: np.ndarray, def_name: str = "x") -> dict:
    """
    Calculates the bounding box of the given points in centroid format.

    :param points: List of points.
    :param def_name: Name of the bounding box.

    :return: Dictionary representing the bounding box.
    """

    # calculate the width, height, and depth of the bounding box
    width = np.max(points[:, 0]) - np.min(points[:, 0])
    depth = np.max(points[:, 1]) - np.min(points[:, 1])
    height = np.max(points[:, 2]) - np.min(points[:, 2])

    # calculate centre of the bounding box
    centroid_x = np.min(points[:, 0]) + width / 2
    centroid_y = np.min(points[:, 1]) + depth / 2
    centroid_z = np.min(points[:, 2]) + height / 2

    # add some padding so the small boxes are visible
    width += 0.2
    depth += 0.2
    height += 0.2

    # convert the centroid and dimensions to floats
    centroid = map(float, (centroid_x, centroid_y, centroid_z))
    dimensions = map(float, (width, depth, height))

    # return the centroid and dimensions of the bounding box
    bounding_box = {
        "name": def_name,
        "centroid": dict(zip(["x", "y", "z"], centroid)),
        "dimensions": dict(zip(["length", "width", "height"], dimensions)),
        "rotations": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    return bounding_box


def get_bbox_edges(bbox_dict: dict) -> np.ndarray:
    """
    Returns the edges of the given bounding box.

    :param bbox_dict: Dictionary containing the bounding box data.

    :return: Numpy array containing the edges of the bounding box.
    """

    x_min = bbox_dict["centroid"]["x"] - bbox_dict["dimensions"]["length"] / 2
    x_max = bbox_dict["centroid"]["x"] + bbox_dict["dimensions"]["length"] / 2
    y_min = bbox_dict["centroid"]["y"] - bbox_dict["dimensions"]["width"] / 2
    y_max = bbox_dict["centroid"]["y"] + bbox_dict["dimensions"]["width"] / 2
    z_min = bbox_dict["centroid"]["z"] - bbox_dict["dimensions"]["height"] / 2
    z_max = bbox_dict["centroid"]["z"] + bbox_dict["dimensions"]["height"] / 2
    edges = ((x_min, y_min, z_min), (x_max, y_max, z_max))

    return np.array(edges)


def get_contained_points_mask(bbox_dict: dict, points: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask for the points that are contained in the given bounding box.

    :param bbox_dict: Dictionary containing the bounding box data.
    :param points: Numpy array containing the points coordinates.

    :return: Boolean mask for the points that are contained in the given bounding box.
    """
    bbox_edges = get_bbox_edges(bbox_dict)

    mask = (points[:, :3] >= bbox_edges[0]) & (points[:, :3] <= bbox_edges[1])
    mask = np.logical_and.reduce(mask, axis=1)

    return mask
