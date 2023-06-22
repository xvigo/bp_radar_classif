"""
Author: Vilem Gottwald

Module for analyzing the parsed frame times.
"""

from datetime import timedelta
import numpy as np
from .dateformat import dt2str


def datatime_differences(frame_times: dict) -> np.ndarray:
    """
    Calculate the differences between the timestamps of the frames.

    :param frame_times: Dict mapping frame numbers to their timestamps.

    :return: Numpy array of differences between the timestamps of the frames.
    """
    differences = []
    outliers = []

    for frame, time in frame_times.items():

        try:
            prev_time = frame_times[frame - 1]
        except KeyError:
            continue

        diff = time - prev_time
        if not (timedelta(milliseconds=49) < diff < timedelta(milliseconds=51)):
            outliers.append(
                [
                    frame - 1,
                    prev_time,
                    frame,
                    time,
                ]
            )

        differences.append(diff.total_seconds())

    differences_np = np.array(differences)
    outliers_np = np.array(outliers)

    print(f"Max difference: {np.max(differences_np)}")
    print(f"Min difference: {np.min(differences_np)}")
    print(f"Mean difference: {np.mean(differences_np)}")

    print(outliers_np)
    print(len(outliers_np))

    return differences_np, outliers_np


def find_missing(frame_nums: list, frame_times: dict) -> list:
    """
    Find missing frames in the dataset.

    :param frame_nums: List of frame numbers.
    :param frame_times: Dict mapping frame numbers to their timestamps.

    :return: List of tuples of missing frame ranges.
    """
    missing_ranges = []
    missing_start = None

    for i in range(min(frame_nums), max(frame_nums) + 1):
        # First Missing
        if i not in frame_nums and missing_start is None:
            missing_start = i

        # First Found
        elif i in frame_nums and missing_start is not None:
            end = i - 1
            cnt = end - missing_start + 1

            T1 = dt2str(frame_times[missing_start - 1]).split("T")[1]
            T2 = dt2str(frame_times[end + 1]).split("T")[1]

            print(f"Missing range: {missing_start} - {end}, {T1} - {T2}, {cnt} missing")
            missing_ranges.append((missing_start, end))
            missing_start = None

    # Enf of list and missing
    if missing_start is not None:
        print(
            f"Missing range: {missing_start} - {end}, {end - missing_start + 1} missing"
        )
        missing_ranges.append((missing_start, end))

    return missing_ranges
