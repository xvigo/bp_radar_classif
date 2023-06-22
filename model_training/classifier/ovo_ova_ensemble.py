"""
Author: Vilem Gottwald

Module containing the one vs one  and one vs all joint classification ensemble.
"""


import numpy as np
import os

from .one_vs_all import Ensemble_ova
from .one_vs_one import Ensemble_ovo
from .generic_ensemble import Ensemble_generic


class Ensemble_ovo_ova(Ensemble_generic):
    """Combination of one vs one and one vs all classifier ensemble"""

    def __init__(self, classes=None):
        """Initialize the ensemble"""
        self.ova = Ensemble_ova()
        self.ovo = Ensemble_ovo()
        super().__init__()
        self.classes = classes

    def load_data(
        self,
        X,
        y,
        split_ratio=0.2,
        verbose=False,
        classes=None,
        split_idx=None,
        scaler_path=None,
    ):
        """Load data and split it into training and validation sets

        :param X: data
        :param y: labels
        :param split_ratio: ratio of validation data
        :param verbose: print info about data
        :param classes: classes to use
        :param split_idx: index of the first sample in the validation set
        :param scaler_path: path to the scaler to normalize the data

        :return: None
        """
        if classes is None:
            classes = self.classes

        self.ova.load_data(
            X,
            y,
            split_ratio=split_ratio,
            verbose=verbose,
            classes=classes,
            split_idx=split_idx,
            scaler_path=scaler_path,
        )
        self.ovo.load_data(
            X,
            y,
            split_ratio=split_ratio,
            verbose=False,
            classes=classes,
            split_idx=split_idx,
            scaler_path=scaler_path,
        )
        self.classes = self.ova.classes
        self.train_X = self.ova.train_X
        self.val_X = self.ova.val_X
        self.train_y = self.ova.train_y
        self.val_y = self.ova.val_y

    def train(
        self,
        get_model_func,
        epochs=60,
        batch_size=256,
        verbose=1,
        save_path=None,
        no_class_weight=False,
        log_dir=None,
        load=False,
    ):
        """Train the ensemble

        :param get_model_func: function returning the model
        :param epochs: number of epochs
        :param batch_size: batch size
        :param verbose: print info about training
        :param save_path: path to save the models
        :param no_class_weight: do not use class weights
        :param log_dir: path to save the logs

        :return: None
        """
        # check if the save path already contains the ensemble name

        # if save_path is not None and os.listdir(save_path) and not load:
        #     raise ValueError(
        #         "The model save path already contains models, which would be overwritten by the new trained models."
        #     )

        self.ovo.train(
            get_model_func,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            save_path=save_path,
            no_class_weight=no_class_weight,
            log_dir=log_dir,
        )

        self.ova.train(
            get_model_func,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True,
            save_path=save_path,
            no_class_weight=no_class_weight,
            log_dir=log_dir,
        )

    def load_models(self, path):
        """Load the models

        :param path: path to the models directory

        :return: True if successful, False otherwise
        """
        return self.ova.load_models(path) and self.ovo.load_models(path)

    def save_models(self, path):
        """Save the models

        :param path: path to the models directory

        :return: True if successful, False otherwise
        """

        return self.ova.save_models(path) and self.ovo.save_models(path)

    def validation_predictions(self, classes=False):
        """Get the validation predictions

        :param classes: return classes instead of probabilities

        :return: predictions
        """
        return self.predict(self.val_X, classes=classes)

    def predict(self, X, classes=False, normalize=False):
        """Predict the classes

        :param X: data
        :param classes: return classes instead of probabilities

        :return: predictions
        """
        ova_pred = self.ova.predict(X)
        ovo_pred = self.ovo.predict(X)
        predictions = self.combine_predicitons(ovo_pred, ova_pred)

        if normalize:
            predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]

        if classes:
            predictions = np.argmax(predictions, axis=1)

        return predictions

    def combine_predicitons(self, ovo_pred, ova_pred):
        """Combine the predictions from one vs one and one vs all classifiers

        :param ovo_pred: one vs one predictions
        :param ova_pred: one vs all predictions

        :return: combined predictions
        """
        y_x = None
        for i in range(len(self.classes)):
            sum = np.zeros((ovo_pred.shape[0]))
            for j in range(len(self.classes)):
                # skip the same class
                if i == j:
                    continue
                ovo_ij = ovo_pred[:, self.ovo.get_selector(i, j)]
                ova_ij = ova_pred[:, i]
                sum += ovo_ij * ova_ij

            y_x = sum[:, np.newaxis] if y_x is None else np.c_[y_x, sum]

        return y_x
