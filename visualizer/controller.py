"""
Author: Vilem Gottwald

Module containing the visualizer controller.
"""
import os

from .model import Model
from .view import View


class Visualizer:
    def __init__(
        self,
        pcd_dir,
        image_dir,
        gt_dir,
        pred_dir,
        animation=False,
        joint_classes=False,
        test_only=False,
    ):
        """Initialize the visualizer

        :param pcd_dir: path to the directory with pointclouds
        :param image_dir: path to the directory with images
        :param gt_dir: path to the directory with ground truth
        :param pred_dir: path to the directory with predictions
        :param animation: if True, the visualizer will animate the visualization
        :param joint_classes: if True, the visualizer will show the joint classes
        :param test_only: if True, the visualizer will show only the test dataset

        :return: None
        """
        self.model = Model(
            pcd_dir,
            image_dir,
            gt_dir,
            pred_dir,
            joint_classes=joint_classes,
            test_only=test_only,
        )
        self.view = View(self)

        if animation:
            self.view.window.set_on_tick_event(self.on_tick_event)

    def _on_next_pcd_button(self):
        """Callback for the next pcd button

        :return: None
        """
        self.model.go_to_next()
        self.update_scenes()

    def _on_prev_pcd_button(self):
        """Callback for the previous pcd button

        :return: None
        """
        self.model.go_to_previous()
        self.update_scenes()

    def _on_jump_to_pcd_button(self):
        """Callback for the jump to pcd button

        :return: None
        """
        self.model.jump_to_index(self.view._jump_to_pcd_idx.int_value)
        self.update_scenes()

    def on_tick_event(self):
        """Callback for the tick event

        :return: None
        """
        self.model.go_to_next()
        self.update_scenes(False)
        self.view.window.post_redraw()

    def update_scenes(self, reset_view=True):
        """Update the scenes

        :param reset_view: if True, the camera will be reset

        :return: None
        """
        pcd_info = self.model.get_current_pcd_info()
        self.view.set_current_pcd_info(*pcd_info)

        image = self.model.get_current_image()
        self.view.set_image(image)

        pcd = self.model.get_current_pcd()
        bboxes, labels = self.model.get_current_gt()
        self.view.update_scene(
            pcd, bboxes, labels, "ground_truth", reset_camera=reset_view
        )

        bboxes, labels, tracks = self.model.get_current_pred()
        self.view.update_scene(
            pcd, bboxes, labels, "prediction", tracks=tracks, reset_camera=reset_view
        )

    def run(self):
        """Run the visualizer

        :return: None
        code"""
        self.update_scenes()
        self.view.run()
