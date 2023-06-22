# Vehicle Classification Using Radar

Author: Vilem Gottwald

The goal of this work is to recognize vehicles from radar point clouds. The radar produces the distance and angle for each target. This representation can be converted into the Cartesian coordinate system to obtain a point cloud 3D representation of the scene. In this thesis, existing approaches to object recognition in point clouds are presented. The method chosen for this thesis consists of object detection using point clustering and subsequent classification using a recurrent neural network. The objects are created from the point clouds using a modified DBSCAN algorithm. Features are extracted from each entity and utilized for classification into different types of vehicles using long short-term memory (LSTM) neural network. A dataset containing 57 345 annotated objects was created to train and evaluate the model. The developed model achieved an F1-score of 83 % on this data.

## Installation

There are two types of installation. First, the lighter installation contains only the dependencies for the visualizer. Second, there is a complete installation of all the dependencies for all the stages, including requirements for the model training.

The code was tested using Python 3.9.16 on WSL2 Ubuntu 22.04.

## Visualizer installation

The requirements for the visualizer are located in the `visualizer_requirements.txt` file.

They can be installed directly using the following:

```
pip install -r visualizer_requirements.txt
```

### Usage

The visualizer can then be executed as follows to show the 3 class results:

```
python -m visualizer --classes 3
```

or the following way to visualize the more detailed vehicle recognition model results:

```
python -m visualizer --classes 5
```

and optionally with a `--test_only` option that shows only the frames from the testing dataset.

## Complete installation

The complete requirements are listed in `complete_requirements.txt`.

They can be installed using the following:

```
pip install -r complete_requirements.txt
```

After that, the models can be trained with custom training parameters using the scripts in the `model_training` directory and also evaluated and visualized.

## Code description

The source code root folder contains 5 directories:

* `/dataset_creation` -  the creation of the dataset, the whole process is step by step implemented in the `convertor.ipynb` file.
* `/detection` - the implementation of the clustering algorithm used for object detection is located in `clusterer.py`. Predicting objects from point clouds using this algorithm is implemented in `predict_detections.py` and evaluation of its performance in `IOU_calculator.py`.
* `/model_training` - models used for the classification of the detected objects. The `classifier` package contains the feature extraction and classification ensembles source code. Training of the models is located inside the Python scripts starting with `train_`. Evaluation of the whole vehicle recognition model is implemented in `evaluate.ipynb`.
* `/visualizer` - this package contains the visualizer of the model results. It can be run using `python -m visualizer --classes n` with `n` argument value `3` for visualizing the results of the model with joint (car/van) and (box truck/truck) classes or `5` for visualizing the results of the model with all classes.
* `/data` - contains the data created by all stages of this thesis including the trained models.

## Data structure

The data directory contains subdirectories for each development step:

* `/parsing` - parsed radar point and targets
* `/labeling` - labels, point clouds and images used for labeling
* `/dataset` - dataset created from the labeled data and json files describing its structure
* `/detections` - objects instances formed in the point clouds by the clustering detection algorithm
* `/training` - saved trained models and a normalization scaler
* `/results` - predictions made by the vehicle recognition model for visualization