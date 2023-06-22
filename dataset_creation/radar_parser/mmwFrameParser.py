"""
Author: Vilem Gottwald

Module for parsing mmwave radar binary files in TLV format.
"""

import struct
from .named_tuples import *
from .bytes_reader import BytesReader
from typing import List


class RadarFrameParser:
    """Parser for frames of mmwave radar binary files in TLV format."""

    # Magic word marking start of another TLV
    MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

    # Format of the TLV headers
    FRAME_HEADER_FORMAT = "=QLQLLLLHH"
    FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FORMAT)

    # TLV formats
    TLV_HEADER_FORMAT = "=II"
    TLV_HEADER_SIZE = struct.calcsize(TLV_HEADER_FORMAT)

    POINT_TYPE = 100
    POINT_FORMAT = "=4fL2h"
    POINT_SIZE = struct.calcsize(POINT_FORMAT)

    TARGET_TYPE = 200
    TARGET_FORMAT = "=L12f2HL"
    TARGET_SIZE = struct.calcsize(TARGET_FORMAT)

    STAT_TYPE = 800
    STATS_FORMAT = "=4f2L"
    STATS_SIZE = struct.calcsize(STATS_FORMAT)

    def __init__(self):
        self.frame_header = None

        self.point_cloud = []
        self.targets = []
        self.stats = []

    @staticmethod
    def _checksum_valid(header: bytes) -> bool:
        """Validates frame header checksum.

        :param header: Frame header bytes.
        :return: True if checksum is valid.
        """
        h = struct.unpack("H" * (len(header) // 2), header)
        a = sum(h)  # & 0xFFFFFFFF
        b = (a >> 16) + (a & 0xFFFF)
        ch_sum = (~b) & 0xFFFF

        return ch_sum == 0

    def parse_header(self, header_bytes: bytes) -> FrameHeader:
        """Parses header bytes and returns it as FrameHeader.

        :param header_bytes: Bytes of the frame header.
        :return: FrameHeader named tuple.
        :raises: ValueError if parsing failed.
        """
        # Check magic bytes
        magic_bytes = header_bytes[0:8]
        if magic_bytes != self.MAGIC_WORD:
            raise ValueError(
                "No correct SYNC pattern (magic word) at the beginning of the header"
            )

        # Check number of bytes of header
        if len(header_bytes) != self.FRAME_HEADER_SIZE:
            raise ValueError("Part of header is missing - not enough bytes")

        # Verify checksum
        if not self._checksum_valid(header_bytes):
            raise ValueError("Header checksum is wrong (does not match calculated)")

        # Unpack to FrameHeader
        header = FrameHeader(*struct.unpack(self.FRAME_HEADER_FORMAT, header_bytes))
        self.frame_header = header

        return header

    def parse_payload(self, payload_bytes: bytes, tlv_cnt: int) -> None:
        """Parses frame payload and stores the results in the parser.

        :param payload_bytes: Bytes representing sequence of TLVs.
        :param tlv_cnt: Number of TLVs inside payload.
        :return: None
        """
        self._clear()
        payload_reader = BytesReader(payload_bytes)
        # Parse each TLV inside the frame
        for n_tlv in range(tlv_cnt):
            # Get TLV header bytes
            if self.TLV_HEADER_SIZE > payload_reader.bytes_left():
                raise ValueError(
                    f"TLV header bytes missing: expected {self.TLV_HEADER_SIZE} bytes,"
                    f" received {payload_reader.bytes_left()} bytes."
                )
            tlv_head_bytes = payload_reader.get_next(self.TLV_HEADER_SIZE)

            # Convert TLV header bytes to named tuple
            tlv_header = TlvHeader(
                *struct.unpack(self.TLV_HEADER_FORMAT, tlv_head_bytes)
            )

            # Check whether remaining payload contains enough bytes for TLV
            remaining_payload_bytes = payload_reader.bytes_left()
            if tlv_header.length > remaining_payload_bytes:
                raise ValueError(
                    f"TLV length ({tlv_header.length}) is invalid - not enough bytes"
                    f" in frame ({remaining_payload_bytes})."
                )

            if tlv_header.type == self.POINT_TYPE:
                # Check if number of points matches the length
                if tlv_header.length % self.POINT_SIZE > 0:
                    print("Point count mismatch", tlv_header.length, self.POINT_SIZE)
                points_cnt = tlv_header.length // self.POINT_SIZE

                self._parse_points(payload_reader, points_cnt)

            elif tlv_header.type == self.TARGET_TYPE:
                # Check if number of targets matches the length
                unexpected_bytes_cnt = tlv_header.length % self.TARGET_SIZE
                if unexpected_bytes_cnt > 0:
                    print(
                        f"Targets count mismatch: {unexpected_bytes_cnt} unexpected bytes."
                    )
                targets_cnt = tlv_header.length // self.TARGET_SIZE

                self._parse_targets(payload_reader, targets_cnt)

            elif tlv_header.type == self.STAT_TYPE:
                self._parse_stats(payload_reader)

            else:
                print(
                    f"  Frame payload error: Unknown TLV type ({tlv_header.type}), len({tlv_header.length})"
                )
                payload_reader.skip_bytes(tlv_header.length)

    def _parse_points(self, payload_reader: BytesReader, points_cnt: int) -> None:
        """Parses point cloud from payload bytes.

        :param payload_reader: BytesReader object.
        :param points_cnt: Number of points to parse.
        :return: None
        """
        for n_point in range(points_cnt):
            point_bytes = payload_reader.get_next(self.POINT_SIZE)
            point = Point(
                *struct.unpack(self.POINT_FORMAT, point_bytes),
                self.frame_header.frameNumber,
            )

            self.point_cloud.append(point)  # append to point cloud

    def _parse_targets(self, payload_reader: BytesReader, targets_cnt: int) -> None:
        """Parses targets from payload bytes.

        :param payload_reader: BytesReader object.
        :param targets_cnt: Number of targets to parse.
        :return: None
        """
        for n_target in range(targets_cnt):
            target_bytes = payload_reader.get_next(self.TARGET_SIZE)
            target = Target(
                *struct.unpack(self.TARGET_FORMAT, target_bytes),
                self.frame_header.frameNumber,
            )

            self.targets.append(target)  # append to target list

    def _parse_stats(self, payload_reader: BytesReader) -> None:
        """Parses stats from payload bytes.

        :param payload_reader: BytesReader object.
        :return: None
        """
        stats_bytes = payload_reader.get_next(self.STATS_SIZE)
        stats = Stats(*struct.unpack(self.STATS_FORMAT, stats_bytes))

        self.stats.append(stats)

    def get_point_cloud(self) -> List[Point]:
        """Returns current point cloud.

        :return: List of Point objects.
        """
        return self.point_cloud

    def get_targets(self) -> List[Target]:
        """Returns current targets.

        :return: List of Target objects.
        """
        return self.targets

    def get_stats(self) -> List[Stats]:
        """Returns current stats.

        :return: List of Stats objects.
        """
        return self.stats

    def _clear(self) -> None:
        """Clears all data from the parser.

        :return: None
        """
        self.point_cloud = []
        self.targets = []
        self.stats = []
