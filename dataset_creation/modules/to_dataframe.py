"""
Author: Vilem Gottwald

Module for converting parsed data to pandas dataframes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import math
import os


def create_dataframe(
    np_points: np.ndarray, np_targets: np.ndarray, frame_times: dict, save_dirpath: str
) -> tuple:
    """
    Create a dataframe from the parsed data.
    Points are converted to cartesian coordinates in the radar view and filtered.

    :param np_points: Numpy array of points.
    :param np_targets: Numpy array of targets.
    :param frame_times: Dictionary of frame times.
    :param save_dirpath: Path to directory where the dataframe will be saved.

    :return: Dataframes of points and targets.
    """

    df_points = pd.DataFrame(
        np_points,
        columns=[
            "range",
            "azimuth",
            "elevation",
            "doppler",
            "targetID",
            "snr",
            "noise",
            "frame",
        ],
    )
    df_targets = pd.DataFrame(
        np_targets,
        columns=[
            "tid",
            "posX",
            "posY",
            "posZ",
            "velX",
            "velY",
            "velZ",
            "accX",
            "accY",
            "accZ",
            "dimX",
            "dimY",
            "dimZ",
            "pointCount",
            "tickCount",
            "state",
            "frame",
        ],
    )

    # Get cartesian coordinates of points in radar view
    elevation_tilt = -6  # vertical radar rotation in degrees
    xz_flip = -1  # flip due to radar being upside down
    radar_height = 6  # radar height in meters
    elevation_tilt_rad = math.radians(elevation_tilt)

    df_points["azimuth"] *= xz_flip
    df_points["elevation"] *= xz_flip

    df_points["x"] = (
        df_points["range"]
        * np.cos(df_points["elevation"])
        * np.sin(df_points["azimuth"])
    )
    df_points["y"] = (
        df_points["range"]
        * np.cos(df_points["elevation"])
        * np.cos(df_points["azimuth"])
    )
    df_points["z"] = df_points["range"] * np.sin(df_points["elevation"])

    # Rotate radar view to world view
    df_points["y"] = df_points["y"] * np.cos(elevation_tilt_rad) - df_points[
        "z"
    ] * np.sin(elevation_tilt_rad)
    df_points["z"] = df_points["y"] * np.sin(elevation_tilt_rad) + df_points[
        "z"
    ] * np.cos(elevation_tilt_rad)

    # Convert targets to world view
    df_targets["posX"] = xz_flip * df_targets["posX"]
    df_targets["posZ"] = xz_flip * df_targets["posZ"]

    # Move radar from 0 up to 6 so 0 represents road surface
    df_targets["posZ"] += radar_height
    df_points["z"] += radar_height

    # Points filtering
    print("Points filtering:")
    prev_cnt = df_points.shape[0]
    orig_cnt = prev_cnt
    print(f"  Original number of points:         {df_points.shape[0]:7}")

    # Remoeve malformed points, ID > 255
    df_points = df_points[(df_points["targetID"] <= 255.0)]
    print(
        f"  After targetID based filtering:     {df_points.shape[0]:7} points, removed: {prev_cnt - df_points.shape[0]:7} points"
    )
    prev_cnt = df_points.shape[0]

    # x-coordinate based clipping
    df_points = df_points[(df_points["x"] < 10.0) & (df_points["x"] > -10.0)]
    print(
        f"  After x-coordinate based filtering: {df_points.shape[0]:7} points, removed: {prev_cnt - df_points.shape[0]:7} points"
    )
    prev_cnt = df_points.shape[0]

    # y-coordinate based clipping
    df_points = df_points[(df_points["y"] < 85.0) & (df_points["y"] > 0.0)]
    print(
        f"  After y-coordinate based filtering: {df_points.shape[0]:7} points, removed: {prev_cnt - df_points.shape[0]:7} points"
    )
    prev_cnt = df_points.shape[0]

    # Z coordinate based clipping
    df_points = df_points[(df_points["z"] < 6.0) & (df_points["z"] > 0.0)]
    print(
        f"  After z-coordinate based filtering: {df_points.shape[0]:7} points, removed: {prev_cnt - df_points.shape[0]:7} points"
    )
    prev_cnt = df_points.shape[0]

    # remove points with opposite doppler speed direction nad point exactly in 0.0
    df_points = df_points[
        ((df_points["x"] > 0) & (df_points["doppler"] > 0))
        | ((df_points["x"] < 0) & (df_points["doppler"] < 0))
    ]
    print(
        f"  After velocity based filtering:     {df_points.shape[0]:7} points, removed: {prev_cnt - df_points.shape[0]:7} points"
    )

    print(
        f"  Final number of points:             {df_points.shape[0]:7} points, overall removed: {orig_cnt - prev_cnt:7} points"
    )

    # Convert to correct data types
    df_points[["frame", "targetID", "snr", "noise"]] = df_points[
        ["frame", "targetID", "snr", "noise"]
    ].astype(int)

    # Add real velocity that is based on the predicted movement direction
    df_points["velocity"] = df_points["doppler"] / (
        np.cos(df_points["elevation"]) * np.cos(df_points["azimuth"])
    )

    # Add corresponding frame_timestamps
    df_points["timestamp"] = df_points["frame"].map(frame_times)

    # Add point unique identification to each point
    df_points["idx"] = df_points.groupby("frame").cumcount()
    df_points["pointID"] = df_points["idx"] + df_points["frame"] * 1000

    # Auxiliary columns used in processing
    df_points["y_orig"] = df_points["y"]
    df_points["total_seconds"] = (
        (df_points["timestamp"] - datetime(2023, 1, 1)).dt.total_seconds().round(3)
    )

    # Save dataframes for later loading
    if save_dirpath is not None:
        df_points.to_pickle(os.path.join(save_dirpath, "radar_points.pkl"))
        df_targets.to_pickle(os.path.join(save_dirpath, "radar_targets.pkl"))

    return df_points, df_targets
