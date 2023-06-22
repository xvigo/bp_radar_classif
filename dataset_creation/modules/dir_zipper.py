"""
Author: Vilem Gottwald

Module for zipping directories.
"""


import os
import zipfile


def zip_dir(src_dir: str, zip_file_path: str) -> None:
    """Zip a directory and all its contents.

    :param src_dir: The directory to zip.
    :param zip_file_path: The path to the zip file.
    :return: None
    """
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(
                    file_path, os.path.relpath(file_path, os.path.join(src_dir, ".."))
                )
