import os

from pathlib import Path
import argparse

# Path to project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.relative_to(Path.cwd())

# Path to data directory
DATA_PATH = PROJECT_ROOT / "data"


# Parse arguments
parser = argparse.ArgumentParser(description="Recognition results visualizer")
parser.add_argument(
    "--classes",
    type=int,
    choices=[3, 5],
    help="Number of classes of the desired used model (3 or 5)",
)
parser.add_argument(
    "--test_only",
    action="store_true",
    help="Show only the frames from the test dataset",
)
parser.add_argument(
    "--not_tracked", action="store_true", help="Show results of untracked frames models"
)
args = parser.parse_args()

# Use the parsed arguments
if args.classes == 3:
    if args.not_tracked:
        PRED_DIR = str(DATA_PATH / "results" / "3class_ensemble_single")
    else:
        PRED_DIR = str(DATA_PATH / "results" / "3class_ensemble_tracked")
    joint_classes = True

elif args.classes == 5:
    if args.not_tracked:
        PRED_DIR = str(DATA_PATH / "results" / "5class_ensemble_single")
    else:
        PRED_DIR = str(DATA_PATH / "results" / "5class_ensemble_tracked")
    joint_classes = False
else:
    print("Error: Invalid number of classes. Please enter 3 or 5.")
    exit()

from .controller import Visualizer

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

IMG_DIR = str(DATA_PATH / "labeling" / "frames_images")
PCD_DIR = str(DATA_PATH / "labeling" / "pointclouds")
GT_DIR = str(DATA_PATH / "dataset" / "dataset_gt")

visualiser = Visualizer(
    PCD_DIR,
    IMG_DIR,
    GT_DIR,
    PRED_DIR,
    joint_classes=joint_classes,
    test_only=args.test_only,
    animation=False,
)

visualiser.run()
