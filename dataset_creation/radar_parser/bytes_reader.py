"""
Author: Vilem Gottwald

Module for reading bytes from a payload.
"""


class BytesReader:
    """Reads bytes from a payload"""

    def __init__(self, payload_bytes: bytes) -> None:
        """Initializes the BytesReader with the payload bytes

        :param payload_bytes: payload bytes
        :return: None
        """
        self.payload_bytes = payload_bytes
        self.offset = 0

    def get_next(self, num_of_bytes: int) -> bytes:
        """Returns next num_of_bytes from the payload

        :param num_of_bytes: number of bytes to return
        :return: next num_of_bytes from the payload
        """
        desired_bytes = self.payload_bytes[self.offset : self.offset + num_of_bytes]
        self.offset += num_of_bytes
        return desired_bytes

    def skip_bytes(self, num_of_bytes: int) -> None:
        """Skips num_of_bytes in the payload

        :param num_of_bytes: number of bytes to skip
        :return: None
        """
        self.offset += num_of_bytes

    def bytes_left(self) -> int:
        """Returns number of bytes left in the payload

        :return: number of bytes left in the payload"""
        return len(self.payload_bytes) - self.offset
