import os

import numpy as np
"""
Author: Vilem Gottwald

Module containing the one vs one classification ensemble.
"""


import tensorflow as tf
from .generic_ensemble import Ensemble_generic


class Ensemble_ova(Ensemble_generic):
    def load_data(
        self,
        X,
        y,
        split_ratio=0.2,
        shuffle=False,
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
        if classes is None:
            self.classes = list(np.unique(y))
        else:
            self.classes = classes

        super().load_data(X, y, split_ratio, shuffle, verbose, split_idx, scaler_path)

    def get_classes_data(self, one_class):
        """Get data for two classes for training and validation sets

        :param one_class: class to use vs all other classes
        """
        train_y_encoded = np.where(self.train_y == one_class, 1, 0)
        val_y_encoded = np.where(self.val_y == one_class, 1, 0)

        return self.train_X, train_y_encoded, self.val_X, val_y_encoded

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
        :param verbose: whether to print info about training
        :param save_path: path to save the models
        :param no_class_weight: whether to use class weights
        :param log_dir: directory to save logs

        :return: None
        """
        for c in self.classes:
            # check if model is already saved and load it
            model_name = f"model_ova_{c}_vs_all"

            if save_path is not None:
                if verbose > 0:
                    print("_" * 120)
                    print(f"Loading classifier for classes {c} vs all from {save_path}")

                if super().load_model(save_path, model_name):
                    continue

            # select the samples for the two classes
            X_train, y_train, X_val, y_val = self.get_classes_data(c)

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

        :param dest_dirpath: directory to save the models

        :return: True if successful, False otherwise
        """
        if not os.path.exists(dest_dirpath):
            os.makedirs(dest_dirpath)

        try:
            for i, model in enumerate(self.classifiers):
                filepath = os.path.join(dest_dirpath, f"model_ova_{i}_vs_rest.h5")
                model.save(filepath)
        except:
            return False
        return True

    def load_models(self, models_dirpath):
        """Load all models from directory

        :param models_dirpath: directory to load the models from

        :return: True if successful, False otherwise
        """

        if not os.path.exists(models_dirpath):
            return False

        # get all model paths
        models = sorted(
            [
                file
                for file in os.listdir(models_dirpath)
                if file.startswith("model_ova")
            ]
        )
        models_paths = [os.path.join(models_dirpath, name) for name in models]

        if not models_paths:
            return False

        # get class combinations from model names
        self.classes = [int(model.split("_")[2]) for model in models]

        # load models
        for model_path in models_paths:
            self.classifiers.append(tf.keras.models.load_model(model_path))

        return True

    def predict(self, X, classes=False):
        """Predict classes for the given data

        :param X: data
        :param classes: whether to return classes or probabilities
        """
        y_pred_all = np.empty((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            # make predictions with the binary classifier
            y_pred = self.classifiers[i].predict(X).flatten()

            # add the predictions to the array and reshape to 2D
            y_pred = y_pred
            y_pred_all[:, i] = y_pred

        # argmax across columns
        if classes:
            y_pred_all = np.argmax(y_pred_all, axis=1)
        # return probabilities
        return y_pred_all
