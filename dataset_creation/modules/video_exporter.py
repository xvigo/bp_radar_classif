"""
Author: Vilem Gottwald

Module for exporting individual frames of video as images.
"""

import datetime
import os.path
import cv2
from .dateformat import str2dt


class VideoFrameExporter:
    """
    Class for exporting individual frames of video as images.
    Saves frame whose timestamp is the closest to given timestamp.
    Processes video frame by frame - timestamp cannot be smaller than its predecessor.
    """

    def __init__(self, filepath: str) -> None:
        """
        Opens video file given by filepath and loads first frame.

        :param filepath: File to open
        :raises errors of the cv2.VideoCapture constructor
        """
        self.filepath = filepath
        self.finished = False

        self.video_capture = cv2.VideoCapture(filepath)

        self.video_FPS = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_frame_duration = datetime.timedelta(seconds=(1 / self.video_FPS))

        video_name = os.path.basename(filepath).rsplit(".", 1)[0]
        self.first_frame_time = str2dt(video_name)

        self.frame_time = self.first_frame_time
        success, self.frame = self.video_capture.read()

        if not success:
            self.finished = True
            raise IOError(f"Couldn't read any frames from {self.filepath}.")

    def __del__(self) -> None:
        """
        Releases video capture.
        """
        self.video_capture.release()

    def export(self, timestamp: datetime.datetime, dst_filepath: str) -> bool:
        """
        Saves frame whose timestamp is the closest to given timestamp as image.

        :param timestamp: Frame with this timestamp should be ideally saved.
        :param dst_filepath: Filepath where the image is stored.
        :return: None - no corresponding image, False - end of video, True - OK
        :raises: IOError if saving exported image failed
        """

        # If video ended don't export any image
        if self.finished:
            return False

        # If video didn't start yet don't export any image
        if timestamp + self.video_frame_duration / 2 < self.first_frame_time:
            print(
                f"Desired frame at {str(timestamp)} is before video start at {str(self.first_frame_time)} - export skipped."
            )
            return None

        # If step backwards don't export any image
        if timestamp + self.video_frame_duration / 2 < self.frame_time:
            print(
                f"Step backwards from {str(self.frame_time)} to {str(timestamp)} - export skipped."
            )
            return None

        # Skip frames and stop at frame right in front of timestamp
        while self.frame_time < timestamp - self.video_frame_duration:
            read_successful = self.read_next_frame()
            if not read_successful:
                return False

        # Next frame is closer than current frame, skip current
        if self.is_next_frame_closer(timestamp):
            read_successful = self.read_next_frame()
            if not read_successful:
                return False

        # Save current frame
        write_successful = cv2.imwrite(dst_filepath, self.frame)
        if not write_successful:
            raise IOError(f"Couldn't write exported image to {dst_filepath}.")

        return True

    def is_next_frame_closer(self, timestamp: datetime.datetime) -> bool:
        """
        Checks whether given timestamp is closer to the next video frame timestamp than current.

        :param timestamp: Timestamp to check.
        :return: True if closer to the next, otherwise False.
        """
        return (timestamp - self.frame_time) > (
            self.frame_time + self.video_frame_duration - timestamp
        )

    def read_next_frame(self) -> bool:
        """
        Reads next frame from video.

        :return: True if frame was read, otherwise False.
        """

        read_success, self.frame = self.video_capture.read()
        self.frame_time += self.video_frame_duration

        if not read_success:
            self.finished = True
            return False

        return True
