"""
Author: Vilem Gottwald

Module containing the one vs one classification ensemble.
"""


import os
from itertools import combinations

import numpy as np
import tensorflow as tf
from .generic_ensemble import Ensemble_generic


class Ensemble_ovo(Ensemble_generic):
    """One vs One classifier ensemble"""

    def load_data(
        self,
        X,
        y,
        split_ratio=0.2,
        shuffle=True,
        verbose=False,
        classes=None,
        split_idx=None,
        scaler_path=None,
    ):
        """Load data and split it into training and validation sets

        :param X: data
        :param y: labels
        :param split_ratio: ratio of validation data
        :param shuffle: shuffle data before splitting
        :param verbose: print info about data
        :param classes: classes to use
        :param split_idx: index of the first sample in the validation set
        :param scaler_path: path to the scaler to normalize the data

        :return: None
        """

        # Set class combinations
        if classes is None:
            classes = list(np.unique(y))
        self.classes = list(combinations(classes, 2))

        super().load_data(X, y, split_ratio, shuffle, verbose, split_idx, scaler_path)

    def select_classes(self, X, y, class1, class2):
        """Get data for the two given classes

        :param X: data
        :param y: labels
        :param class1: first class
        :param class2: second class
        :return: data and labels for the two classes
        """

        # Select training data for binary classification of two classes
        classes_mask = np.where(np.logical_or(y == class1, y == class2))
        X_binary = X[classes_mask]
        y_binary = y[classes_mask]
        y_binary = np.where(y_binary == class1, 1, 0)

        return X_binary, y_binary

    def get_classes_data(self, class1, class2):
        """Get data for two classes for training and validation sets

        :param class1: first class
        :param class2: second class

        :return: data and labels for the two classes
        """
        bin_train_X, bin_train_y = self.select_classes(
            self.train_X,
            self.train_y,
            class1,
            class2,
        )
        bin_val_X, bin_val_y = self.select_classes(
            self.val_X,
            self.val_y,
            class1,
            class2,
        )

        return bin_train_X, bin_train_y, bin_val_X, bin_val_y

    def train(
        self,
        get_model_func,
        epochs=100,
        batch_size=256,
        verbose=1,
        save_path=None,
        no_class_weight=False,
        log_dir=None,
    ):
        """Train classifiers for all class combinations

        :param get_model_func: function to get the model
        :param epochs: number of epochs
        :param batch_size: batch size
        :param verbose: print info about training
        :param save_path: path to save the models
        :param no_class_weight: do not use class weights
        :param log_dir: path to save the logs

        :return: None
        """
        for c in self.classes:
            # check if model is already saved and load it
            model_name = f"model_ovo_{c[0]}_vs_{c[1]}"

            if save_path is not None:
                if verbose > 0:
                    print("_" * 120)
                    print(
                        f"Loading classifier for classes {c[0]} and {c[1]} from {save_path}"
                    )

                if super().load_model(save_path, model_name):
                    continue

            # select the samples for the two classes
            X_train, y_train, X_val, y_val = self.get_classes_data(c[0], c[1])

            super().train(
                X_train,
                y_train,
                X_val,
                y_val,
                model_name,
                get_model_func,
                epochs,
                batch_size,
                verbose,
                save_path,
                no_class_weight,
                log_dir,
            )

    def save_models(self, dest_dirpath):
        """Save all models to directory

        :param dest_dirpath: path to the directory to save the models

        :return: True if successful, False otherwise
        """
        for i, model in enumerate(self.classifiers):
            cls_comb = self.classes[i]
            model_name = f"model_ovo_{cls_comb[0]}_{cls_comb[1]}.h5"
            if not super().save_model(model, dest_dirpath, model_name):
                return False

        return True

    def load_models(self, models_dirpath):
        """Load all models from directory

        :param models_dirpath: path to the directory with models

        :return: True if successful, False otherwise
        """
        if not os.path.exists(models_dirpath):
            return False

        # get all model paths
        models = [
            file for file in os.listdir(models_dirpath) if file.startswith("model_ovo")
        ]
        models.sort()
        models_paths = [os.path.join(models_dirpath, name) for name in models]

        if not models_paths:
            return False

        self.classes = []
        # get class combinations from model names
        for model in models:
            parts = model.rsplit(".", 1)[0].split("_")
            class_comb = (int(parts[2]), int(parts[4]))
            self.classes.append(class_comb)

        # load models
        for model_path in models_paths:
            self.classifiers.append(tf.keras.models.load_model(model_path))
        return True

    def get_selector(self, i, j):
        """Get index of the classifier outputs for the given class combination i vs j

        :param i: first class
        :param j: second class

        :return: index of the probability of first class vs the second class in the predictions
        """

        if i == j:
            raise ValueError("Invalid indices, classifying same class")

        # 5 classes ensemble
        if len(self.classifiers) == 10:
            mapper = [
                [None, 0, 2, 4, 6],
                [1, None, 8, 10, 12],
                [3, 9, None, 14, 16],
                [5, 11, 15, None, 18],
                [7, 13, 17, 19, None],
            ]

        # 3 classes ensemble
        elif len(self.classifiers) == 3:
            mapper = [
                [None, 0, 2],
                [1, None, 4],
                [3, 5, None],
            ]
        else:
            raise NotImplementedError

        return mapper[i][j]

    def predict(self, X):
        """Predict class for each sample

        :param X: data to predict on

        :return: predicted class probabilities for each sample
        """
        y_pred_all = np.empty((X.shape[0], len(self.classes), 2))

        for i, c in enumerate(self.classes):
            # make predictions with the binary classifier
            y_pred = self.classifiers[i].predict(X).flatten()

            y_pred_all[:, i, 0] = y_pred
            y_pred_all[:, i, 1] = 1 - y_pred

        # return the most probbable class for each sample
        y_pred_all = y_pred_all.reshape((X.shape[0], -1))

        return y_pred_all
