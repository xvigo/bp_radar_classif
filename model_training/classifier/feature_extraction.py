"""
Author: Vilem Gottwald

Module for extracting features from dataset.
"""

import os
import numpy as np

# print numpy arrays without scientific notation
np.set_printoptions(suppress=True)


class FeaturesExtractor:
    """Class for extracting features for object classification"""

    # dictionary of columns in numpy array dataset
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
    }

    def __init__(
        self,
        points_threshold: int = 4,
        num_timesteps: int = 7,
        num_features: int = 37,
        columns_dict: dict = None,
    ) -> None:
        """Creates new dataset extractor instance

        :param points_threshold: minimum number of points in object
        :param num_timesteps: number of timesteps in object
        :param num_features: number of features in object
        """

        self.num_features = num_features
        self.num_timesteps = num_timesteps

        self.points_threshold = points_threshold

        self.objects_features = np.array([])
        self.objects_classes = np.array([])
        self.object_ids = np.array([])

        if columns_dict is not None:
            self._IDX = columns_dict

    def extract_from_dataset_gt(self, dataset_dirpath) -> tuple:
        """Extracts features and classes from dataset

        :param dataset_dirpath: path to dataset directory
        :return: tuple of features and classes
        """

        if not os.path.isdir(dataset_dirpath):
            raise ValueError("Dataset directory does not exists")

        dataset_filepaths = sorted(
            [os.path.join(dataset_dirpath, f) for f in os.listdir(dataset_dirpath)]
        )

        objects_features = []
        objects_classes = []

        for data_file in dataset_filepaths:
            # Load single frame points
            points = np.load(
                data_file
            )  # x, y, z, snr, noise, velocity, class_id, obj_id,

            # Extract object ids
            object_ids = np.unique(points[:, self._IDX["object_id"]])

            # Extract features and class for each object
            for object_id in object_ids:
                object_points = points[points[:, self._IDX["object_id"]] == object_id]

                # If object has less than 4 points ..skip it
                if object_points.shape[0] < self.points_threshold:
                    continue

                features = self.extract_features(object_points)
                objects_features.append(features)
                objects_classes.append(object_points[0, self._IDX["class_id"]])

        self.objects_classes = np.array(objects_classes, dtype="i")
        self.objects_features = np.array(objects_features)

        return (self.objects_features, self.objects_classes)

    def extract_from_dataset_pred(self, dataset_dirpath) -> tuple:
        """Extracts features and classes from dataset

        :param dataset_dirpath: path to dataset directory

        :return: tuple of features and object ids
        """

        if not os.path.isdir(dataset_dirpath):
            raise ValueError("Dataset directory does not exists")

        dataset_filepaths = sorted(
            [os.path.join(dataset_dirpath, f) for f in os.listdir(dataset_dirpath)]
        )

        objects_features = []
        object_ids = []

        for data_file in dataset_filepaths:
            # Load single frame points
            points = np.load(
                data_file
            )  # x, y, z, snr, noise, velocity, class_id, obj_id,

            # Extract object ids
            frame_objects_ids = np.unique(points[:, self._IDX["object_id"]])

            # Extract features and class for each object
            for object_id in frame_objects_ids:
                object_points = points[points[:, self._IDX["object_id"]] == object_id]

                # If object has less than 4 points ..skip it
                if object_points.shape[0] < self.points_threshold:
                    continue

                features = self.extract_features(object_points)
                objects_features.append(features)
                object_ids.append(object_id)

        self.object_ids = np.array(object_ids, dtype="i")
        self.objects_features = np.array(objects_features)

        return (self.objects_features, self.object_ids)

    def load_from_saved_gt(self, features_path, classes_path) -> bool:
        """Loads dataset features and classes from previsouly saved featured features as numpy files

        :param features_path: path to features file
        :param classes_path: path to classes file

        :return: object features and classes
        """

        if not os.path.isfile(features_path) or not os.path.isfile(classes_path):
            raise FileNotFoundError("Features or classes file does not exists")

        # if both files exists load them
        self.objects_features = np.load(features_path)
        self.objects_classes = np.load(classes_path)

        return self.objects_features, self.objects_classes

    def load_from_saved_pred(self, features_path, object_ids_path) -> bool:
        """Loads dataset features and classes from previsouly saved featured features as numpy files

        :param features_path: path to features file
        :param classes_path: path to classes file

        :return: object features and ids
        """

        if not os.path.isfile(features_path) or not os.path.isfile(object_ids_path):
            raise FileNotFoundError("Features or object_ids file does not exists")

        # if both files exists load them
        self.objects_features = np.load(features_path)
        self.object_ids = np.load(object_ids_path)

        return self.objects_features, self.object_ids

    def save_gt(self, features_path, classes_path) -> bool:
        """Saves dataset features and classes to numpy files

        :param features_path: path to features file
        :param classes_path: path to classes file
        :return: True if saving was successful, otherwise False
        """
        if self.objects_classes.size and self.objects_features.size:
            np.save(features_path, self.objects_features)
            np.save(classes_path, self.objects_classes)
            return True

        return False

    def save_pred(self, features_path, object_ids_path) -> bool:
        """Saves dataset features and classes to numpy files

        :param features_path: path to features file
        :param classes_path: path to classes file
        :return: True if saving was successful, otherwise False
        """
        if self.object_ids.size and self.objects_features.size:
            np.save(features_path, self.objects_features)
            np.save(object_ids_path, self.object_ids)
            return True

        return False

    def get_gt(self, three_classes=False) -> tuple:
        """Returns dataset features and classes

        :return: tuple of features and classes
        """
        if three_classes:
            objects_classes_joint = np.copy(self.objects_classes)
            objects_classes_joint[self.objects_classes == 2] = 1
            objects_classes_joint[self.objects_classes == 3] = 2
            objects_classes_joint[self.objects_classes == 4] = 2
            return (self.objects_features, objects_classes_joint)

        return (self.objects_features, self.objects_classes)

    def get_pred(self) -> tuple:
        """Returns dataset features and classes

        :return: tuple of features and classes
        """
        return (self.objects_features, self.object_ids)

    def extract_features(self, object_points):
        """Extracts features from object points

        :param object_points: object points
        :return: features
        """
        # swap y and y_orig
        object_points[:, [self._IDX["y_orig"], self._IDX["y"]]] = object_points[
            :, [self._IDX["y"], self._IDX["y_orig"]]
        ]

        # Calculate the number of extracted features and create an empty np array for them
        n_inf = 6
        feat_p_inf = 6
        features = np.zeros((self.num_timesteps, self.num_features))

        sliding_window_width = 0.150
        stride = 0.06
        start_time = object_points[:, self._IDX["total_seconds"]].max() - 0.5

        time_idx = 0
        # For each time step in the frame
        for time_window_idx in range(self.num_timesteps):
            # Calculate time window
            start_timewindow = start_time + time_window_idx * stride
            end_timewindow = start_timewindow + sliding_window_width

            # Select points in time window
            time_step_points = object_points[
                (object_points[:, self._IDX["total_seconds"]] >= start_timewindow)
                & (object_points[:, self._IDX["total_seconds"]] <= end_timewindow)
            ]

            # If no points in time window ..skip it
            if time_step_points.shape[0] == 0:
                continue

            # Extract from each points property
            for info_idx in range(n_inf):
                offset = info_idx * feat_p_inf
                selected_information = time_step_points[:, info_idx]

                # Min, Max, Mean, Spread, StdDev, Var
                features[time_idx, offset] = selected_information.min()
                features[time_idx, offset + 1] = selected_information.max()
                features[time_idx, offset + 2] = selected_information.mean()
                features[time_idx, offset + 3] = abs(np.ptp(selected_information))
                features[time_idx, offset + 4] = np.std(selected_information)
                features[time_idx, offset + 5] = np.var(selected_information)

            # Number of points
            features[time_idx, n_inf * feat_p_inf] = time_step_points.shape[0]

            time_idx += 1
        return np.array(features)

    @staticmethod
    def load_gt_classes(path, join_classes=False):
        """Loads classes from numpy file

        :param path: path to numpy file
        :param join_classes: if True, similar classes are joined into one class

        :return: classes
        """
        classes = np.load(path)

        if join_classes:
            # Join classes 1, 2 and 3, 4 into 1 and 2
            classes[classes == 2] = 1
            classes[classes == 3] = 2
            classes[classes == 4] = 2

        return classes
