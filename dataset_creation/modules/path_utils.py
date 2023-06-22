"""
Author: Vilem Gottwald

Module for path utilities.
"""

import os


def listdir_paths(directory: str) -> list:
    """
    Returns list of paths to files in given directory.

    :param directory: Path to directory

    :return: List of paths to files in given directory
    """
    return sorted([os.path.join(directory, file) for file in os.listdir(directory)])
