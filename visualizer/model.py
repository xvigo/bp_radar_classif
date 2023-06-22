"""
Author: Vilem Gottwald

Module containing the visualizer model.
"""
import json
import os
import open3d as o3d
from pyntcloud import PyntCloud
import numpy as np
import pkg_resources
from datetime import datetime


def listdir_fullpaths(dir_path):
    """Returns a sorted list of full paths to all files in the given directory"""
    return sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)])


def str2dt(str_timestamp):
    """Convert string with format %Y%m%dT%H%M%S%f to datetime and return it.

    :param str_timestamp: string with format %Y%m%dT%H%M%S%f
    :return: datetime object
    """
    return datetime.strptime(str_timestamp, "%Y%m%dT%H%M%S%f")


class Model:
    # index of each column in the results np array
    _IDX = {
        "x": 0,
        "y": 1,
        "z": 2,
        "snr": 3,
        "noise": 4,
        "velocity": 5,
        "y_orig": 6,
        "total_seconds": 7,
        "point_id": 8,
        "object_id": 9,
        "class_id": 10,
        "IOU": 11,
    }

    track_colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (0.5, 0.5, 0.0),  # Olive
        (0.5, 0.0, 0.5),  # Purple
        (0.0, 0.5, 0.5),  # Teal
        (1.0, 0.5, 0.0),  # Orange
        (0.5, 1.0, 0.0),  # Lime
        (0.0, 0.5, 1.0),  # Sky blue
        (1.0, 0.0, 0.5),  # Rose
        (0.5, 0.0, 1.0),  # Violet
        (0.0, 1.0, 0.5),  # Spring green
        (1.0, 0.5, 0.5),  # Light pink
        (0.5, 1.0, 0.5),  # Light green
        (0.5, 0.5, 1.0),  # Light blue
        (1.0, 1.0, 0.5),  # Pale yellow
        (1.0, 0.5, 1.0),  # Light magenta
        (0.5, 1.0, 1.0),  # Light cyan
        (0.7, 0.7, 0.7),  # Gray
        (0.3, 0.3, 0.3),  # Dark gray
        (0.9, 0.0, 0.0),  # Dark red
        (0.0, 0.9, 0.0),  # Dark green
        (0.0, 0.0, 0.9),  # Dark blue
        (0.9, 0.9, 0.0),  # Dark yellow
        (0.9, 0.0, 0.9),  # Dark magenta
        (0.0, 0.9, 0.9),  # Dark cyan
    ]

    def __init__(
        self, pcd_dir, image_dir, gt_dir, pred_dir, joint_classes=False, test_only=False
    ):
        """Initializes the model

        :param pcd_dir: Path to the directory containing the point cloud files
        :param image_dir: Path to the directory containing the images
        :param gt_dir: Path to the directory containing the ground truth files
        :param pred_dir: Path to the directory containing the prediction files
        :param IOU_frames: List of frames to show IOU for
        :param joint_classes: Whether to show all classes or only the joint classes
        """
        self.pcd_dir = pcd_dir
        self.pcd_filepaths = listdir_fullpaths(pcd_dir)

        if test_only:
            self.pcd_filepaths = self.pcd_filepaths[6211:8240]

        self.max_index = len(self.pcd_filepaths) - 1

        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

        self.joint_classes = joint_classes
        if joint_classes:
            self.pred_id2classname = {0: "noise", 1: "car", 2: "truck"}
            self.gt_id2classname = {
                0: "noise",
                1: "car",
                2: "car",
                3: "truck",
                4: "truck",
            }
        else:
            self.id2classname = {
                0: "noise",
                1: "car",
                2: "van",
                3: "box_truck",
                4: "truck",
            }

        self.current_index = 0

        self.color_pallete = np.load(
            pkg_resources.resource_filename(__name__, "color_pallete.npy")
        )

        self.pred_pos_history = dict()
        self.id_track_colors = dict()
        self.color_idx = 0

    def jump_to_index(self, index):
        """Sets the current index to the given index

        :param index: The index to jump to
        """
        self.current_index = index
        if index > self.max_index:
            self.current_index = self.max_index
        elif index < 0:
            self.current_index = 0

        self.pred_pos_history.clear()

    def go_to_next(self):
        """Increments the current index by one"""
        self.current_index = (
            self.current_index + 1
            if self.current_index < self.max_index
            else self.max_index
        )

    def go_to_previous(self):
        """Decrements the current index by one"""
        self.current_index = self.current_index - 1 if self.current_index > 0 else 0

        self.pred_pos_history.clear()

    def get_curr_filepath(self, type):
        """Returns the path to the current file of the given type

        :param type: The type of file to get the path for
        :return: The path to the file
        """
        filepath = self.pcd_filepaths[self.current_index]
        name = os.path.basename(filepath).rsplit(".", 1)[0]
        if type == "pcd":
            return filepath
        elif type == "img":
            return os.path.join(self.image_dir, name + ".jpg")
        elif type == "gt":
            return os.path.join(self.gt_dir, name + ".npy")
        elif type == "pred":
            return os.path.join(self.pred_dir, name + ".npy")

    def get_current_pcd_info(self):
        """Returns the current point cloud as an open3d PointCloud object

        :return: The current point cloud information
        """
        basename = os.path.basename(self.get_curr_filepath("pcd")).split(".")[0]
        return str2dt(basename), self.current_index, self.max_index

    def get_current_pcd(self):
        """Returns the current point cloud as an open3d PointCloud object"""
        pcd_path = self.get_curr_filepath("pcd")
        return self.get_points_from_file(pcd_path)

    def get_current_image(self):
        """Returns the current image as an open3d Image object"""
        img_path = self.get_curr_filepath("img")
        return self.get_image_from_file(img_path)

    def get_current_gt(self):
        """Returns the current ground truth bounding boxes as a list of AxisAlignedBoundingBox objects"""
        gt_path = self.get_curr_filepath("gt")
        return self.get_bounding_boxes_from_file(gt_path)

    def get_current_pred(self):
        """Returns the current predicted bounding boxes as a list of AxisAlignedBoundingBox objects"""
        gt_path = self.get_curr_filepath("pred")
        return self.get_bounding_boxes_from_file(gt_path, track=True, gt=False)

    def get_points_from_file(self, filename):
        """Reads point cloud from file and returns it as an open3d PointCloud object

        :param filename: The path to the file to read
        :return: The point cloud as an open3d PointCloud object
        """
        # pcd = o3d.io.read_point_cloud(filename)
        try:
            pcd_df = PyntCloud.from_file(filename)
        except FileNotFoundError:
            return None

        colors = self._colorize_points(pcd_df.points["velocity"].to_numpy())
        points = pcd_df.points[["x", "y", "z"]].to_numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def get_image_from_file(self, filename):
        """Reads image from file and returns it as an open3d Image object

        :param filename: The path to the file to read
        :return: The image as an open3d Image object
        """
        try:
            image = o3d.io.read_image(filename)
        except FileNotFoundError:
            # empty image
            image = o3d.geometry.Image()
        return image

    def get_color(self, class_id):
        """Returns the color for the given class id

        :param class_id: The class id to get the color for
        :return: The color for the given class id
        """
        if class_id not in self.id_track_colors:
            color = self.track_colors[self.color_idx]
            self.id_track_colors[class_id] = color
            self.color_idx = (self.color_idx + 1) % len(self.track_colors)
        else:
            color = self.id_track_colors[class_id]

        return color

    def update_tracks(self, object_ids, bboxes):
        """Updates the current tracks with the given tracks

        :param object_ids: The object ids of the tracks
        :param bboxes: The bounding boxes of the tracks
        """
        centers = []
        for bbox in bboxes:
            center = bbox.get_center()
            # object is on the left side of the road
            if center[0] > 0:
                center[1] = bbox.get_min_bound()[1]
            # object is on the right side of the road
            else:
                center[1] = bbox.get_max_bound()[1]

            centers.append(center)

        track_points = []
        track_indices = []
        track_colors = []

        for i, (object_id, center) in enumerate(zip(object_ids, centers)):

            # update position history
            if object_id not in self.pred_pos_history:
                self.pred_pos_history[object_id] = [center]
            else:
                self.pred_pos_history[object_id].append(center)

            object_positions = self.pred_pos_history[object_id]
            if len(object_positions) > 1:
                offset = len(track_points)

                indices = [
                    [i + offset, i + offset + 1]
                    for i in range(len(self.pred_pos_history[object_id]) - 1)
                ]
                track_points.extend(self.pred_pos_history[object_id])
                track_indices.extend(indices)
                track_colors.extend([self.get_color(object_id)] * len(indices))

        if track_points == []:
            return None

        tracks = o3d.geometry.LineSet()
        tracks.points = o3d.utility.Vector3dVector(track_points)
        tracks.lines = o3d.utility.Vector2iVector(track_indices)
        tracks.colors = o3d.utility.Vector3dVector(track_colors)
        return tracks

    def get_bounding_boxes_from_file(self, filename, track=False, gt=True):
        """Reads bounding boxes from file and returns them as a list of AxisAlignedBoundingBox objects

        :param filename: The path to the file to read
        :param track: Whether to show the tracks of the bounding boxes

        :return: The bounding boxes as a list of AxisAlignedBoundingBox objects
        """
        if not self.joint_classes:
            labels_dict = self.id2classname
        elif gt:
            labels_dict = self.gt_id2classname
        else:
            labels_dict = self.pred_id2classname

        bboxes = []
        classes_ids = []
        try:
            points = np.load(filename)
        except FileNotFoundError:
            return bboxes, classes_ids

        object_ids = np.unique(points[:, self._IDX["object_id"]]).tolist()
        for object_id in object_ids:
            object_rows = points[points[:, self._IDX["object_id"]] == object_id]
            try:
                object_class = object_rows[0, self._IDX["class_id"]]
            except IndexError:
                object_class = -1

            xmin, ymin, zmin = object_rows[:, :3].min(axis=0) - 0.1
            xmax, ymax, zmax = object_rows[:, :3].max(axis=0) + 0.1
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(xmin, ymin, zmin), max_bound=(xmax, ymax, zmax)
            )
            bboxes.append(bbox)
            classes_ids.append(object_class)

        classes_labels = [labels_dict.get(id, "") for id in classes_ids]

        if track:
            tracks = self.update_tracks(object_ids, bboxes)
            return bboxes, classes_labels, tracks
        else:
            return bboxes, classes_labels

    def _colorize_points(self, velocities):
        """Colors points based on their velocity

        :param velocities: The velocities of the points
        :return: The colors of the points
        """
        palette_len = self.color_pallete.shape[0] - 1
        velocity_diff = (
            velocities.max() - velocities.min()
            if velocities.max() != velocities.min()
            else 1
        )

        colors = np.zeros((velocities.shape[0], 3))

        for idx, value in enumerate(velocities):
            colors[idx] = self.color_pallete[
                round((value - velocities.min()) / (velocity_diff) * palette_len)
            ]
        return colors
