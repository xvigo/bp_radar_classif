"""
Author: Vilem Gottwald

Module containing the visualizer view.
"""

import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np


def centered_label(text, in_vertical=False):
    """Create a label centered in a horizontal layout

    :param text: text to display
    :param in_vertical: if True, the label will be also centered in a vertical layout

    :return: the layout
    """
    horiz = gui.Horiz()
    horiz.add_stretch()
    horiz.add_child(gui.Label(text))
    horiz.add_stretch()

    if in_vertical:
        layout = gui.Vert()
        layout.add_stretch()
        layout.add_child(horiz)
        layout.add_stretch()
    else:
        layout = horiz

    return layout


class View:
    def __init__(self, controller):
        """Initialize the view

        :param controller: the controller
        """
        self.controller = controller

        self.ground_truth_labels = []
        self.predict_labels = []

        # Create the application
        self.app = gui.Application.instance
        self.app.initialize()

        # Create the window
        self.window = self.app.create_window("Detection Visualizer", 1025, 512)

        # Initialize the layout
        self._init_layout()

        # Set the layout callback that will be called when the window is resized
        self.window.set_on_layout(self.set_main_layout)

        # Create material for point cloud
        self.pcd_mat = o3d.visualization.rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = 4

        # Create material for bounding box
        self.bbox_mat = o3d.visualization.rendering.MaterialRecord()
        self.bbox_mat.shader = "unlitLine"
        BLUE_COLOR = (0.0, 62 / 255, 168 / 255, 1.0)
        self.bbox_mat.base_color = BLUE_COLOR
        self.bbox_mat.line_width = 2

        # Create material for road lines
        self.road_lines_mat = o3d.visualization.rendering.MaterialRecord()
        self.road_lines_mat.shader = "unlitLine"
        ROAD_COLOR = (130 / 255, 130 / 255, 130 / 255, 1.0)
        self.road_lines_mat.base_color = ROAD_COLOR
        self.road_lines_mat.line_width = 2

        self.track_lines_mat = o3d.visualization.rendering.MaterialRecord()
        self.track_lines_mat.shader = "unlitLine"
        self.track_lines_mat.line_width = 3

        # Create road lines
        self.road_lines = self._get_road_lines()

        def on_key(e):
            if e.key == gui.KeyName.RIGHT:
                if e.type == gui.KeyEvent.DOWN:  # check UP so we default to DOWN
                    self.controller._on_next_pcd_button()
                return gui.Widget.EventCallbackResult.HANDLED

            elif e.key == gui.KeyName.LEFT:
                if e.type == gui.KeyEvent.DOWN:  # check UP so we default to DOWN
                    self.controller._on_prev_pcd_button()
                return gui.Widget.EventCallbackResult.HANDLED

            return gui.Widget.EventCallbackResult.IGNORED

        self.scene_gtruth.set_on_key(on_key)

    def _init_layout(self):
        """Initialize the layout of the window"""
        # Define the background color for 3D scenes
        GREY_COLOR = (0.4, 0.4, 0.4, 1.0)
        em = self.window.theme.font_size

        # Add header
        self.header_gt = centered_label("Ground truth", in_vertical=True)
        self.window.add_child(self.header_gt)
        self.header_pred = centered_label("Prediction", in_vertical=True)
        self.window.add_child(self.header_pred)
        self.header_img = centered_label("Image", in_vertical=True)
        self.window.add_child(self.header_img)

        # Add the ground truth scene
        self.scene_gtruth = gui.SceneWidget()
        self.scene_gtruth.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer
        )
        self.scene_gtruth.scene.set_background(GREY_COLOR)
        self.window.add_child(self.scene_gtruth)

        # Add the predicted scene
        self.scene_predict = gui.SceneWidget()
        self.scene_predict.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer
        )
        self.scene_predict.scene.set_background(GREY_COLOR)
        self.window.add_child(self.scene_predict)

        ## Add the third column
        self.C3 = gui.Vert()
        self.window.add_child(self.C3)

        horiz_current_timestamp = gui.Horiz()
        horiz_current_timestamp.add_stretch()
        self.pcd_label = gui.Label("                  ")
        horiz_current_timestamp.add_child(self.pcd_label)
        horiz_current_timestamp.add_stretch()

        horiz_current_idx = gui.Horiz()
        horiz_current_idx.add_stretch()
        self.idx_label = gui.Label("                   ")
        slash = gui.Label(" / ")
        self.idx_total = gui.Label("                   ")
        horiz_current_idx.add_child(self.idx_label)
        horiz_current_idx.add_child(slash)
        horiz_current_idx.add_child(self.idx_total)
        horiz_current_idx.add_stretch()

        ### Add the navigation panel
        self.C3.add_child(centered_label("Navigation"))
        self.C3.add_child(horiz_current_timestamp)
        self.C3.add_child(horiz_current_idx)
        self.nav_horiz_layout = gui.Horiz()
        self.C3.add_child(self.nav_horiz_layout)

        self.C3.add_stretch()

        ### Add the image widget
        self.image_widget = gui.ImageWidget()
        self.C3.add_child(self.image_widget)

        #### Add the load next frame button
        # Horizontal navigation layout
        self.nav_horiz_layout.add_stretch()

        # Previous frame button
        self.prev_pcd_button = gui.Button("Previous")
        self.prev_pcd_button.set_on_clicked(self.controller._on_prev_pcd_button)
        self.nav_horiz_layout.add_child(self.prev_pcd_button)

        # Jump to frame button
        self.jump_to_layout = gui.Vert()
        self.nav_horiz_layout.add_child(self.jump_to_layout)

        # Next frame button
        self.next_pcd_button = gui.Button("Next")
        self.next_pcd_button.set_on_clicked(self.controller._on_next_pcd_button)
        self.nav_horiz_layout.add_child(self.next_pcd_button)

        # Jump to frame button container
        self.jump_to_pcd_button = gui.Button("Jump to")
        self.jump_to_pcd_button.set_on_clicked(self.controller._on_jump_to_pcd_button)
        self.jump_to_layout.add_child(self.jump_to_pcd_button)

        self._jump_to_pcd_idx = gui.NumberEdit(gui.NumberEdit.INT)
        self.jump_to_layout.add_child(self._jump_to_pcd_idx)

        self.nav_horiz_layout.add_stretch()

    def set_main_layout(self, layout_context):
        """Set the layout of the window

        :param layout_context: The layout context
        """
        r = self.window.content_rect

        # Set the header layout component dimensions

        header_height = 2 * layout_context.theme.font_size
        scene_height = r.height - header_height
        label_y = r.y + scene_height
        col_width = r.width / 3
        col_x = r.x

        # Set first column frame
        self.header_gt.frame = gui.Rect(col_x, label_y, col_width, header_height)
        self.scene_gtruth.frame = gui.Rect(col_x, r.y, col_width, scene_height)
        col_x += col_width

        # Set second column frame
        self.header_pred.frame = gui.Rect(col_x, label_y, col_width, header_height)
        self.scene_predict.frame = gui.Rect(col_x, r.y, col_width, scene_height)
        col_x += col_width

        # Set third column frame
        self.header_img.frame = gui.Rect(col_x, label_y, col_width, header_height)
        self.C3.frame = gui.Rect(col_x, r.y, col_width, scene_height)

    def _get_scene_widget(self, type):
        """Get the scene widget for the given scene type

        :param type: The scene type
        :return: The scene widget
        """
        if type == "ground_truth":
            return self.scene_gtruth
        elif type == "prediction":
            return self.scene_predict
        else:
            raise ValueError("Unknown scene type: " + type)

    def _get_scene_labels(self, type):
        """Get the scene labels for the given scene type

        :param type: The scene type
        :return: The scene labels
        """
        if type == "ground_truth":
            return self.ground_truth_labels
        elif type == "prediction":
            return self.predict_labels
        else:
            raise ValueError("Unknown scene type: " + type)

    def set_current_pcd_info(self, name, idx, total):
        """Set the current point cloud info

        :param name: The point cloud name
        :param idx: The point cloud index
        :param total: The total number of point clouds
        """
        self.pcd_label.text = f"{name}"
        self.idx_label.text = str(idx).rjust(10)
        self.idx_total.text = str(total).ljust(10)

    def set_image(self, image):
        """Set the image to display

        :param image: The image to display
        """
        self.image_widget.update_image(image)

    def add_points_to_scene(self, pcd, scene):
        """Add the given point cloud to the given scene

        :param pcd: The point cloud to add
        :param scene: The scene to add the point cloud to
        """
        # select scene
        scene_widget = self._get_scene_widget(scene)

        # add point cloud to scene
        scene_widget.scene.add_geometry("point_cloud", pcd, self.pcd_mat)

    def add_bboxes_to_scene(self, bboxes, labels, scene):
        """Add the given bounding boxes to the given scene

        :param bboxes: The bounding boxes to add
        :param labels: The labels of the bounding boxes
        :param scene: The scene to add the bounding boxes to
        """
        # select scene and its current labels
        scene_widget = self._get_scene_widget(scene)
        scene_labels = self._get_scene_labels(scene)

        # add bboxes and their labels to scene
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            scene_widget.scene.add_geometry(f"bounding_box_{i}", bbox, self.bbox_mat)
            label_pos = bbox.get_center()
            label_pos[2] = bbox.get_max_bound()[2] + 1
            label_pos[1] = bbox.get_max_bound()[1] + 1
            label_pos[0] = bbox.get_min_bound()[0]
            label_3d = scene_widget.add_3d_label(label_pos, label)
            label_3d.color = gui.Color(255 / 255, 255 / 255, 255 / 255, 1.0)
            scene_labels.append(label_3d)

    def add_tracks_to_scene(self, tracks, scene):
        """Add the given tracks to the given scene

        :param tracks: The tracks to add
        :param scene: The scene to add the tracks to
        """
        # select scene
        if tracks is None:
            return

        scene_widget = self._get_scene_widget(scene)
        scene_widget.scene.add_geometry("track_lines", tracks, self.track_lines_mat)

    def _remove_scene_labels(self, scene):
        """Remove the labels from the given scene

        :param scene: The scene to remove the labels from
        """
        # select scene and its current labels
        scene_widget = self._get_scene_widget(scene)
        scene_labels = self._get_scene_labels(scene)

        # remove labels from scene
        for label in scene_labels:
            scene_widget.remove_3d_label(label)

        # clear labels list
        scene_labels.clear()

    def clear_scene(self, scene):
        """Clear the given scene

        :param scene: The scene to clear
        """
        scene_widget = self._get_scene_widget(scene)
        scene_widget.scene.clear_geometry()
        self._remove_scene_labels(scene)

    def reset_camera(self, scene):
        """Reset the camera of the given scene

        :param scene: The scene to reset the camera of
        """
        scene_w = self._get_scene_widget(scene)

        scene_w.setup_camera(60, scene_w.scene.bounding_box, (0, 42, 3))
        scene_w.look_at((0, 30, 0), (0, -10, 55), (0, 0, 1))

    @staticmethod
    def _get_road_lines():
        """Get the road lines

        :return: The road lines
        """
        # Create points and indices for road lines
        vertical_points = [[x, y, 0] for x in range(-10, 11, 5) for y in (0, 85)]
        horizontal_points = [[x, y, 0] for y in range(0, 86, 5) for x in (-10, 10)]

        line_points = vertical_points + horizontal_points
        line_indices = [[x, x + 1] for x in range(0, len(line_points), 2)]

        # Create o3d geometry LineSet
        road_lines = o3d.geometry.LineSet()
        road_lines.points = o3d.utility.Vector3dVector(line_points)
        road_lines.lines = o3d.utility.Vector2iVector(line_indices)

        return road_lines

    def add_road_lines_to_scene(self, scene):
        """Add the road lines to the given scene

        :param scene: The scene to add the road lines to
        """
        scene_widget = self._get_scene_widget(scene)
        scene_widget.scene.add_geometry(
            "road_lines", self.road_lines, self.road_lines_mat
        )

    def update_scene(self, pcd, bboxes, labels, scene, tracks=None, reset_camera=True):
        """Update the given scene with the given point cloud, bounding boxes and labels

        :param pcd: The point cloud to add
        :param bboxes: The bounding boxes to add
        :param labels: The labels of the bounding boxes
        :param scene: The scene to add the point cloud to
        :param tracks: The tracks to add
        :param reset_camera: Whether to reset the camera
        """

        self.clear_scene(scene)
        self.add_road_lines_to_scene(scene)
        if pcd is not None:
            self.add_points_to_scene(pcd, scene)

        self.add_bboxes_to_scene(bboxes, labels, scene)
        self.add_tracks_to_scene(tracks, scene)

        if reset_camera:
            self.reset_camera(scene)

    def run(self):
        """Run the application"""
        self.app.run()
