"""
Author: Vilem Gottwald

Module for removing orphan files from directories.
Used to remove frame files for where image, point cloud or annotation files are missing.
"""
import os
import itertools


def remove_files_from_dir(files: str, dir: str) -> None:
    """
    Removes files from given directory.

    :param files: List of filenames to remove
    :param dir: Directory to remove files from

    :return: None
    """
    for file in files:
        file_path = os.path.abspath(os.path.join(dir, file))
        os.remove(file_path)
        print(f"removed {file}")


def get_orphan_files(look_for: str, check_against: str, ext: str = None) -> list:
    """
    Returns list of files in look_for that are not in check_against.

    :param look_for: Directory to look for files in
    :param check_against: Directory to check against
    :param ext: File extension to add to filenames in look_for

    :return: List of filenames in look_for that are not in check_against
    """
    look_for_files = {f.split(".")[0] for f in os.listdir(look_for)}
    check_agains_files = {f.split(".")[0] for f in os.listdir(check_against)}

    orphans = look_for_files.difference(check_agains_files)

    if type(ext) is str:
        orphans = [name + ext for name in orphans]
    print(
        f"Found {len(orphans)} filenames in {look_for} that are not in {check_against}."
    )
    return sorted(orphans)


def remove_all_permutations_orphans(dir1: str, dir2: str, dir3: str) -> None:
    """
    Removes files from dir1 that are not in dir2 or dir3.
    Removes files from dir2 that are not in dir1 or dir3.
    Removes files from dir3 that are not in dir1 or dir2.

    :param dir1: Directory to remove files from and check against
    :param dir2: Directory to remove files from and check against
    :param dir3: Directory to remove files from and check against

    :return: None
    """
    for d1, d2 in itertools.permutations([dir1, dir2, dir3], 2):
        orphans = get_orphan_files(d1, d2)
        remove_files_from_dir(orphans, d1)
