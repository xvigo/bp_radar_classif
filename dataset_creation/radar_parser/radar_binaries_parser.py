"""
Author: Vilem Gottwald

Module for parsing radar recordings binaries.
"""

from .mmwFrameParser import RadarFrameParser
from .file_content import FileContent
import os
import datetime
from .named_tuples import Point, Target, Stats


class RadarRecordingsParser:
    """Class for parsing radar recordings binaries."""

    def __init__(self):
        """Init parser with path to directory containing radar recordings binaries."""
        self.parser = RadarFrameParser()
        self.points_list = []
        self.targets_list = []
        self.stats_list = []
        self.frame_times = {}

    def parse(self, recordings_dir: str) -> None:
        """Parses radar recordings binaries and return points, dataframes and dict mapping frames to timestamps.

        :param recordings_dir: Path to directory containing radar recording files.
        :return: List of points, list of targets and dict mapping frames to their timestamps.
        """

        # List all binary files created by radar
        record_files = [
            os.path.join(recordings_dir, file)
            for file in os.listdir(recordings_dir)
            if file.endswith(".bin")
        ]
        print(
            f"Found {len(record_files)} files containing radar recordings in {recordings_dir}."
        )

        # Process each binary file
        for filepath in record_files:
            print(f"Parsing recording file {os.path.basename(filepath)}...")

            # Init file contents for handling the TLV sequences
            data_file = FileContent(filepath)
            prev_header_timestamp = None

            # Process all frames inside file
            while not data_file.is_empty():

                # Find the start of next frame
                if not data_file.find_pattern(self.parser.MAGIC_WORD):
                    break

                # Read frame header bytes and parse them
                header_bytes = data_file.read(self.parser.FRAME_HEADER_SIZE)
                try:
                    frame_header = self.parser.parse_header(header_bytes)
                except ValueError as err_msg:
                    print(f"  Frame header error: {err_msg}")
                    # skip magic word bytes and continue to locating following one
                    data_file.unread(len(header_bytes) - len(self.parser.MAGIC_WORD))
                    continue

                # Read frame payload bytes and parse them
                frame_payload_len = (
                    frame_header.totalFrameLen - self.parser.FRAME_HEADER_SIZE
                )
                data_bytes = data_file.read(frame_payload_len)
                try:
                    self.parser.parse_payload(data_bytes, frame_header.numTLVs)
                except ValueError as err_msg:
                    print(f"  Frame payload parsing error: {err_msg}")
                    # skip magic word bytes and continue to locating following one
                    data_file.unread(
                        len(data_bytes)
                        + len(header_bytes)
                        - len(self.parser.MAGIC_WORD)
                    )
                    continue

                # Add frame timestamp to dict
                if prev_header_timestamp is None:  # first frame in file
                    frame_time = data_file.get_filename_date()
                else:
                    timestamp_diff = frame_header.timestamp - prev_header_timestamp
                    frame_time += datetime.timedelta(microseconds=timestamp_diff)
                self.frame_times[frame_header.frameNumber] = frame_time
                prev_header_timestamp = frame_header.timestamp

                # Append current frame into list
                self.points_list.append(self.parser.get_point_cloud())
                self.targets_list.append(self.parser.get_targets())
                self.stats_list.append(self.parser.get_stats())

        print("Parsing finished!")

        return self.points_list, self.targets_list, self.frame_times

    def print_stats(self) -> None:
        """Prints stats about the parsed frames.

        Prints the total number of parsed frames and the number of frames with no points, no targets,
        no points no targets, no points some targets, some points no targets and some points some targets.

        :return: None
        """
        print(f"Total number of parsed frames: {len(self.points_list)}")

        no_points = 0
        no_targets = 0
        no_points_some_targets = 0
        some_points_no_targets = 0
        no_points_no_targets = 0
        both = 0

        frames = zip(self.points_list, self.targets_list)
        for frame_points, frame_targets in frames:
            if not len(frame_points):
                no_points += 1

                if len(frame_targets):
                    no_points_some_targets += 1
                else:
                    no_points_no_targets += 1

            if not len(frame_targets):
                no_targets += 1
                if len(frame_points):
                    some_points_no_targets += 1

            if len(frame_points) and len(frame_targets):
                both += 1

        print(f" - No points: {no_points}")
        print(f" - No targets: {no_targets}")
        print(f" - No points no targets: {no_points_no_targets}")
        print(f" - No points some targets: {no_points_some_targets}")
        print(f" - Some points no targets: {some_points_no_targets}")
        print(f" - Some points some targets: {both}")

    def get_points(self) -> list[list[Point]]:
        """Returns list of lists of points from all parsed frames.

        :return: List of lists of points from all parsed frames.
        """
        return self.points_list

    def get_targets(self) -> list[list[Target]]:
        """Returns list of lists of targets from all parsed frames.

        :return: List of lists of targets from all parsed frames.
        """
        return self.targets_list

    def get_stats(self) -> list[Stats]:
        """Returns list of stats from all parsed frames.

        :return: List of stats from all parsed frames.
        """
        return self.stats_list

    def get_frame_times(self) -> dict[int, datetime.datetime]:
        """Returns dict mapping frame numbers to their timestamps.

        :return: Dict mapping frame numbers to their timestamps.
        """
        return self.frame_times
