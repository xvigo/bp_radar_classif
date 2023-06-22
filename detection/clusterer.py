"""
Author: Vilem Gottwald

Module containing the object detection clustering algorithm.
"""

import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from more_itertools import pairwise


class Clusterer:
    """Clusterer class that performs customized DBSCAN clustering on the points in a dataframe."""

    def __init__(
        self,
        y_scale=1.8,
        overlap_lanes=True,
        combine_tall=True,
        split_wide=True,
        track_objects=True,
        dbscan_metric_fcn=None,
    ):
        """Initialize the clusterer with the given parameters.

        :param y_scale: The scale of the y axis. Used to scale the y column before clustering.
        :param overlap_lanes: Whether to overlap the lanes or not.
        :param combine_tall: Whether to combine tall clusters or not.
        :param split_wide: Whether to split wide clusters or not.
        :param track_objects: Whether to track objects or not.
        :param dbscan_metric_fcn: The distance metric function to use for DBSCAN clustering.

        :return: None
        """
        self.y_scale = y_scale
        self.overlap_lanes = overlap_lanes
        self.combine_tall = combine_tall
        self.split_wide = split_wide
        track_objects = track_objects

        self.dbscan_metric_fcn = (
            self.def_dbscan_metric_fnc
            if dbscan_metric_fcn is None
            else dbscan_metric_fcn
        )

        if track_objects:
            self.prev_detections_df = None
            self.max_object_id = -1

    @staticmethod
    def def_dbscan_metric_fnc(p1, p2):
        """
        Custom distance metric that calculates the distance between two points
        in 3D space (x, y, doppler speed) using the Euclidean distance.

        :param p1: First point.
        :param p2: Second point.

        :return: The distance between the two points.
        """
        diff = p1[0:2] - p2[0:2]
        dist = np.sqrt(np.sum(np.square(diff)))

        doppler_diff = abs(p1[2] - p2[2])

        total = dist + doppler_diff * 2
        return total

    @staticmethod
    def objects_share_lane(first_points, second_points):
        """
        Return whether the two objects are in the same lane.

        :param first_points: Numpy array of points of the first object.
        :param second_points: Numpy array of points of the second object.

        :return: Boolean value indicating whether the two objects are in the same lane.
        """
        lane_edges = [(-10.0, -4.05), (-4.05, 0.0), (0.0, 4.35), (4.35, 10.0)]

        first_center = np.mean(first_points)
        second_center = np.mean(second_points)

        first_lane = None
        # get line of teh first bounding box
        for lane in lane_edges:
            if lane[0] < first_center < lane[1]:
                first_lane = lane
                break

        if first_lane is None:
            return False

        # return whether the second bbox center x coordinate is in the same line
        return first_lane[0] < second_center < first_lane[1]

    def cluster(self, points):
        """Cluster the points in the dataframe and return the result as a numpy array.

        :param points: The points to cluster. Can be a pandas dataframe or a numpy array.

        :return: A numpy array of the clustered points.
        """
        # Convert the points to a dataframe
        if isinstance(points, pd.DataFrame):
            df = points.copy()
        elif isinstance(points, np.ndarray):
            df = pd.DataFrame(
                points[:, :6], columns=["x", "y", "z", "snr", "noise", "velocity"]
            ).copy()
        else:
            raise ValueError("points must be a pandas dataframe or a numpy array")

        # Scale the y column
        df["y_scaled"] = df["y"] / self.y_scale

        clustered_df = self.cluster_by_lanes(df)

        if self.overlap_lanes:
            clustered_df = self.unite_overlapping_lanes(clustered_df)

        if self.split_wide:
            clustered_df = self.split_wide_clusters(clustered_df)

        if self.combine_tall:
            clustered_df = self.combine_tall_clusters(clustered_df)

        clustered_df["cluster"] = clustered_df["cluster"].astype("category").cat.codes
        clustered_df = clustered_df.drop(columns=["y_scaled", "index"])

        return clustered_df.to_numpy()

    def split_lanes(self, df):
        """Split the dataframe into 4 lanes dataframes based on the x value and the overlap_lanes flag.

        :param df: The dataframe to split.

        :return: A list of 4 dataframes, one for each lane.
        """
        if self.overlap_lanes:
            ranges = [(-10.0, -4.05), (-5.0, 0.0), (0.0, 5.0), (4.35, 10.0)]
        else:
            ranges = [(-10.0, -4.05), (-4.05, 0.0), (0.0, 4.35), (4.35, 10.0)]

        dfs_lanes = [
            df.loc[(df["x"] >= x_min) & (df["x"] < x_max)].copy()
            for (x_min, x_max) in ranges
        ]

        return dfs_lanes

    def cluster_by_lanes(self, df):
        """Cluster the points in each lane separately and combine the results into one dataframe.

        :param df: The dataframe to cluster.

        :return: A dataframe containing the clustered points.
        """
        dfs_lanes = self.split_lanes(df)

        # Perform DBSCAN clustering on each lane
        max_label = -1
        for i, df in enumerate(dfs_lanes):
            # Skip empty lanes
            if df.empty:
                continue

            # Perform DBSCAN clustering on the x, y_scaled and velocity columns
            np_points = df[["x", "y_scaled", "velocity"]].to_numpy()
            dbscan = DBSCAN(eps=3.4, min_samples=1, metric=self.dbscan_metric_fcn).fit(np_points)  # type: ignore - metrics can also be a callable
            labels = list(dbscan.labels_)

            # Shift labels to differentiate clusters from different lanes
            for idx, label in enumerate(labels):
                # Skip noise points
                if label < 0:
                    continue
                # Shift the label
                labels[idx] += max_label + 1

            max_label = max(max(labels), max_label)

            dfs_lanes[i]["cluster"] = labels

        # Concatenate the lanes back together
        clustered_df = pd.concat(dfs_lanes).reset_index()
        return clustered_df

    def unite_overlapping_lanes(self, clustered_df):
        """Unite clusters that contain similar points - caused by the overlap of the lanes.

        :param clustered_df: The dataframe containing the clustered points.

        :return: A dataframe containing the clustered points with the overlapping clusters united.
        """

        # Get list of tuples of indexes of points that are in the same position in different lanes
        duplicit_indexes = [
            tuple(indexes)
            for indexes in clustered_df.groupby(
                ["x", "y", "z", "snr", "noise", "velocity"]
            ).groups.values()
            if len(indexes) > 1
        ]

        # Unite the clusters
        for (first, second) in duplicit_indexes:
            cluster1 = clustered_df.iloc[first]["cluster"]
            cluster2 = clustered_df.iloc[second]["cluster"]
            clustered_df.loc[clustered_df["cluster"] == cluster1, "cluster"] = cluster2

        # Remove duplicate points created by the unification
        clustered_df = clustered_df.drop_duplicates()

        return clustered_df

    def combine_tall_clusters(self, clustered_df):
        """Combine clusters that are tall and could be trucks split into half.

        :param clustered_df: The dataframe containing the clustered points.

        :return: A dataframe containing the clustered points with the tall clusters combined.
        """
        # Get list of all cluster labels
        cluster_labels = [
            label for label in clustered_df["cluster"].unique() if label > -1
        ]

        # Get the cluster labels of the clusters that are in the most left and right lanes (lanes with the highest truck density)
        cluster_labels_mostright = [
            cluster_label
            for cluster_label in cluster_labels
            if clustered_df[clustered_df["cluster"] == cluster_label]["x"].min() >= 4.35
        ]
        cluster_labels_mostleft = [
            cluster_label
            for cluster_label in cluster_labels
            if clustered_df[clustered_df["cluster"] == cluster_label]["x"].max()
            <= -4.05
        ]

        # Perform the combining for both lanes
        clusters_by_lanes = (cluster_labels_mostleft, cluster_labels_mostright)
        for cluster_labels in clusters_by_lanes:
            # Sort the clusters by their y coordinate
            cluster_labels.sort(
                key=lambda x: clustered_df[clustered_df["cluster"] == x]["y"].min()
            )

            for first_cluster, second_cluster in pairwise(cluster_labels):

                # Get the dataframes of the clusters
                first_df = clustered_df[clustered_df["cluster"] == first_cluster]
                second_df = clustered_df[clustered_df["cluster"] == second_cluster]

                # Get the difference in velocity between the clusters
                velocity_diff = abs(
                    first_df["velocity"].mean() - second_df["velocity"].mean()
                )
                if velocity_diff > 0.5:
                    continue

                # Get the cluster labels of the clusters that are taller than 3.5 meters
                first_cluster_height = first_df["z"].max()
                if first_cluster_height < 3.4:
                    continue

                if (first_df["x"].abs().max() + 1 < second_df["x"].abs().max()) or (
                    first_df["x"].abs().min() - 1 > second_df["x"].abs().min()
                ):
                    continue

                if second_df.shape[0] > 10:
                    continue

                total_len = second_df["y"].max() - first_df["y"].min()
                if total_len > 20:
                    continue

                distance = second_df["y"].min() - first_df["y"].max()
                if distance > 15:
                    continue

                clustered_df.loc[
                    clustered_df["cluster"] == second_cluster, "cluster"
                ] = first_cluster

        return clustered_df

    def split_wide_clusters(self, clustered_df):
        """Split clusters that are wider than 5.6 meters. (reflections and vehicles travelling in parallel)

        :param clustered_df: The dataframe containing the clustered points.
        """
        # Split clusters that are wider than 5 meters
        cluster_labels = [
            label for label in clustered_df["cluster"].unique() if label > -1
        ]

        # Check each clusters width
        for cluster_label in cluster_labels[:]:
            cluster_points_df = clustered_df[clustered_df["cluster"] == cluster_label]
            cluster_width = abs(
                cluster_points_df["x"].max() - cluster_points_df["x"].min()
            )

            split_edge = 4.05 if cluster_points_df["x"].mean() < 0 else 4.35

            # Split bounding boxes wider than 5 meters
            if cluster_width > 4.5:
                # Change the cluster label of the points that are in the left lane to a new cluster
                new_cluster = max(cluster_labels) + 1
                cluster_labels.append(new_cluster)

                clustered_df.loc[
                    (clustered_df["cluster"] == cluster_label)
                    & (clustered_df["x"].abs() <= split_edge),
                    "cluster",
                ] = new_cluster

        return clustered_df

    def unite_tracked(self, clustered_df):
        """Unite clusters that are tracked by containing the same point.

        :param clustered_df: The dataframe containing the clustered points.

        :return: A dataframe containing the clustered points with the tracked clusters united.
        """
        if self.prev_clustered_df is None:
            self.prev_clustered_df = clustered_df
            return clustered_df

        # Shift the current clusters ids so they differ from the previous ones
        clustered_df["cluster"] += self.max_object_id + 10

        # Get the previous and current cluster ids
        prev_clustered_labels = self.prev_clustered_df["cluster"].unique().tolist()
        cluster_labels = clustered_df["cluster"].unique().tolist()

        for cluster_label in cluster_labels:
            cluster_sel = clustered_df[clustered_df["cluster"] == cluster_label]

            for prev_clustered_label in prev_clustered_labels:
                prev_cluster_sel = self.prev_clustered_df[
                    self.prev_clustered_df["cluster"] == prev_clustered_label
                ]

                # if any point in current cluster was in previous cluster copy its label
                if np.any(np.isin(prev_cluster_sel["pointID"], cluster_sel["pointID"])):

                    # if another cluster was already assigned to the previous one
                    old_cluster_sel = clustered_df[
                        clustered_df["cluster"] == prev_clustered_label
                    ]
                    if old_cluster_sel.shape[0] > 0:
                        # Check if the clusters are in the same lane
                        if self.objects_share_lane(
                            cluster_sel["x"],
                            old_cluster_sel["x"],
                        ):
                            # If yes, add them both
                            clustered_df.loc[
                                clustered_df["cluster"] == cluster_label, "cluster"
                            ] = prev_clustered_label

                        # If not, and new has more points remove the old one and add the new
                        elif cluster_sel.shape[0] > old_cluster_sel.shape[0]:
                            # Assign the new cluster to the previous one and remove the old one
                            clustered_df[
                                clustered_df.loc["cluster"] == prev_clustered_label,
                                "cluster",
                            ] = np.inf
                            clustered_df[
                                clustered_df.loc["cluster"] == cluster_label, "cluster"
                            ] = prev_clustered_label
                            clustered_df[
                                clustered_df.loc["cluster"] == np.inf, "cluster"
                            ] = cluster_label

                    else:
                        # First one, add it
                        clustered_df.loc[
                            clustered_df["cluster"] == cluster_label, "cluster"
                        ] = prev_clustered_label
                    break

        self.prev_clustered_df = clustered_df.copy()

        return clustered_df

    def new_sparse_ids_to_dense(self, clustered_df):
        """Convert newly detected objects sparse ids to dense.

        :param clustered_df: The dataframe containing the clustered points.

        :return: A dataframe containing the clustered points with the new ids converted to dense.
        """
        # Get the new object ids

        new_object_ids = np.unique(
            clustered_df.loc[clustered_df["cluster"] > self.max_object_id, "cluster"]
        )

        # Convert
        for unique_object_id in new_object_ids:
            clustered_df.loc[clustered_df["cluster"] == unique_object_id, "cluster"] = (
                self.max_object_id + 1
            )
            self.max_object_id += 1

        return clustered_df

    @staticmethod
    def get_cluster_bounds(cluster_points):
        """Get the bounding box of a cluster of points.

        :param cluster_points: The points of the cluster.

        :return: The bounding box of the cluster.
        """
        x_min = cluster_points[0].min()
        x_max = cluster_points[0].max()
        y_min = cluster_points[1].min()
        y_max = cluster_points[1].max()
        z_min = cluster_points[2].min()
        z_max = cluster_points[2].max()

        return (x_min, x_max, y_min), (y_max, z_min, z_max)
