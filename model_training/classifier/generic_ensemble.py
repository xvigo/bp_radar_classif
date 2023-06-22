"""
Author: Vilem Gottwald

Module containing generic ensemble classifier.
"""


import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

tf.random.set_seed(7)


class Ensemble_generic:
    """Generic ensemble classifier"""

    def __init__(self):
        """Initialize classifier"""
        self.classifiers = []
        self.classes = []
        self.train_X = None
        self.val_X = None
        self.train_y = None
        self.val_y = None

    @staticmethod
    def split_data(data, offset, test_ratio=0.2):
        """Split data into training and validation sets

        :param data: data to split
        :param offset: offset of the first sample in the data
        :param test_ratio: ratio of test data

        :return: tuple of training and validation data
        """
        test_cnt = int(data.shape[0] * test_ratio)
        test_data = data[offset : offset + test_cnt]
        train_data = np.concatenate((data[:offset], data[offset + test_cnt :]), axis=0)
        return train_data, test_data

    def load_data(
        self,
        X,
        y,
        split_ratio=0.2,
        shuffle=False,
        verbose=False,
        split_idx=None,
        scaler_path=None,
    ):
        """Load data and split it into training and validation sets

        :param X: features
        :param y: labels
        :param split_ratio: ratio of validation data
        :param shuffle: whether to shuffle the data
        :param verbose: whether to print info about the data
        :param split_idx: index of the first sample in the validation set
        :param scaler_path: path to the scaler to normalize the data

        :return: None
        """

        if scaler_path is not None:
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        if split_idx is None:
            self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
                X, y, test_size=split_ratio, random_state=42, shuffle=shuffle
            )
        else:
            self.train_X, self.val_X = self.split_data(
                X, split_idx, test_ratio=split_ratio
            )
            self.train_y, self.val_y = self.split_data(
                y, split_idx, test_ratio=split_ratio
            )

        if verbose:
            print("Number of training samples:", len(self.train_X))
            print("Number of testing samples:", len(self.val_X))

    @staticmethod
    def get_class_weights(y_train):
        """ " Get class weights for training

        :param y_train: training labels

        :return: class weights
        """

        print(y_train.shape, y_train)
        class_weights_arr = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(enumerate(class_weights_arr))
        return class_weights

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        model_name: str,
        get_model_func,
        epochs=100,
        batch_size=256,
        verbose=1,
        save_path=None,
        no_class_weight=False,
        log_dir=None,
    ):
        """Train classifiers for all class combinations

        :param X_train: training features
        :param y_train: training labels
        :param X_val: validation features
        :param y_val: validation labels
        :param model_name: name of the model
        :param get_model_func: function to get the model
        :param epochs: number of epochs
        :param batch_size: batch size
        :param verbose: whether to print info about the training
        :param save_path: path to save the model
        :param no_class_weight: whether to use class weights
        :param log_dir: path to the log directory

        :return: None
        """

        # get class weights for training
        class_weights = None if no_class_weight else self.get_class_weights(y_train)

        if verbose > 0:
            print("_" * 120)
            print(f"Training {model_name.replace('_', ' ')}...")

        log_path = os.path.join(log_dir, f"log_{model_name}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_path, histogram_freq=1
        )

        # create the model
        model = get_model_func()
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[tensorboard_callback],
        )

        if save_path is not None:
            self.save_model(model, save_path, model_name)

        # add the trained classifier to the list
        self.classifiers.append(model)

    def save_model(self, model, dest_dirpath, model_name):
        """Save models to directory

        :param model: model to save
        :param dest_dirpath: path to the directory
        :param model_name: name of the model

        :return: True if the model was saved successfully, False otherwise
        """
        if not os.path.exists(dest_dirpath):
            os.makedirs(dest_dirpath)
        try:
            filepath = os.path.join(dest_dirpath, f"{model_name}.h5")
            model.save(filepath)
        except:
            return False

        return True

    def load_model(self, model_dir, model_name):
        """Load all models from directory

        :param model_dir: path to the directory
        :param model_name: name of the model

        :return: True if the model was loaded successfully, False otherwise

        """

        model_path = os.path.join(model_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            return False

        try:
            model = tf.keras.models.load_model(model_path)
            self.classifiers.append(model)
        except:
            return False

        return True

    def prob_to_class(self, probabilities):
        """Convert probabilities to class labels

        :param probabilities: probabilities of each class

        :return: class labels
        """
        return (probabilities > 0.5).astype("int")

    def predict(self, X, classes=False):
        """Predict class labels for the given data

        :param X: data to predict
        :param classes: whether to return class labels or probabilities
        """
        raise NotImplementedError

    def validation_predictions(self, classes=False):
        """Predict class labels for the validation data

        :param classes: whether to return class labels or probabilities

        :return: class labels or probabilities
        """
        return self.predict(self.val_X, classes=classes)

    def metrics(self, train_data=False):
        """Calculate and print metrics train or validation data

        :param train_data: whether to calculate metrics for training data

        :return: confusion matrix
        """
        if train_data:
            prediction = self.predict(self.train_X, classes=True)
            truth = self.train_y
        else:
            prediction = self.validation_predictions(classes=True)
            truth = self.val_y

        confusion_mtx = confusion_matrix(truth, prediction)

        print(f"Accuracy:  {accuracy_score(truth, prediction) * 100:0.2f} %")
        print(
            f"Precision: {precision_score(truth, prediction, average='macro') * 100:0.2f} %"
        )
        print(
            f"Recall:    {recall_score(truth, prediction, average='macro') * 100:0.2f} %"
        )
        print(f"F1-score:  {f1_score(truth, prediction, average='macro') * 100:0.2f} %")
        print(f"\nConfusion matrix:\n", confusion_mtx)

        return confusion_mtx
