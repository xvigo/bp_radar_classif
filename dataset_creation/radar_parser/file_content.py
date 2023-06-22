"""
Author: Vilem Gottwald

Module for loading and reading file contents.
"""

import os.path
from modules import dateformat


class FileContent:
    """Class for loading and reading file contents."""

    def __init__(self, filepath: str) -> None:
        """Opens file and loads its contents.

        :param filepath: File to open and load its contents
        :raises FileNotFoundError: if file does not exist
        """
        self.filepath = filepath
        self.byte_ptr = 0

        with open(filepath, "rb") as data_file:
            self.data = data_file.read()

        self.total_length = len(self.data)

        # Get date from filename in format "<date>(_drop)?.bin"
        date_str = os.path.basename(filepath).rsplit(".", 1)[0].rsplit("_", 1)[0]
        self.filename_date = dateformat.str2dt(date_str)

    def read(self, size: int) -> bytes:
        """Reads given number of bytes from file contents and shifts current byte pointer forward.

        :param size: number of bytes to read
        :return: bytes that were read
        """
        ret_data = self.data[self.byte_ptr : self.byte_ptr + size]

        self.byte_ptr += size
        if self.byte_ptr > self.total_len():
            self.byte_ptr = self.total_len()

        return ret_data

    def unread(self, size: int) -> None:
        """Unreads given number of bytes from file contents - shifts current byte pointer backwards.

        :param size: Number of bytes to unread
        """
        self.byte_ptr -= size
        if self.byte_ptr < 0:
            self.byte_ptr = 0

    def find_pattern(self, sync_pattern: bytes) -> bool:
        """Shifts current byte pointer until sync_pattern is found.

        :param sync_pattern: pattern to match.
        :return: if found True, otherwise False.
        """
        if self.is_empty():
            return False

        try:
            pattern_start = self.data[self.byte_ptr :].index(sync_pattern)
        except ValueError:
            self.byte_ptr = self.total_length
            return False

        self.byte_ptr += pattern_start
        return True

    def set_position(self, index: int) -> None:
        """Set current position in file contents.

        :param index: Position index 0 - file contents length
        """
        if index < 0 or index > self.total_len():
            raise ValueError(
                f"Position index {index} out of bounds <0, {len(self.data)}>."
            )
        self.byte_ptr = index

    def get_filepath(self) -> str:
        """Returns filepath of the loaded file.

        :return: Filepath
        """
        return self.filepath

    def get_filename(self) -> str:
        """Returns filename of the loaded file.

        :return: Filename
        """
        return os.path.basename(self.filepath)

    def get_filename_date(self) -> str:
        """Returns datetime that was parsed from filename.

        :return: Datetime
        """
        return self.filename_date

    def __len__(self) -> int:
        """Returns length of file contents after any reading.

        :return: Remaining length of file contents
        """

        return len(self.data) - self.byte_ptr

    def total_len(self) -> int:
        """Returns total length of file contents before any reading.

        :return: Total length of file contents
        """
        return self.total_length

    def is_empty(self) -> bool:
        """Returns bool whether whole content was read already.

        :return: True if EOF, otherwise false
        """
        return self.byte_ptr >= self.total_len()
