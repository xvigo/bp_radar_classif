"""
Author: Vilem Gottwald

Training of the 3 class OvO OvA ensemble.

Note: Jupyter Notebook wasn't used because the kernel sometimes crashed during the long training.      
"""

from common import (
    DATA_PATH,
    SCALER_PATH,
    DATASET_SPLIT_IDX,
    DATASET_PATH,
    FEATURES_PATH,
    CLASSES_PATH,
)
from classifier import FeaturesExtractor
from keras.layers import LSTM, Dense, Masking
from keras.models import Sequential
from keras.metrics import Precision, Recall
from classifier import Ensemble_ovo_ova


# Load features and gt classes from dataset
dataset_extractor = FeaturesExtractor()
try:
    dataset_extractor.load_from_saved_gt(FEATURES_PATH, CLASSES_PATH)
except FileNotFoundError:
    dataset_extractor.extract_from_dataset_gt(DATASET_PATH)
    dataset_extractor.save_gt(FEATURES_PATH, CLASSES_PATH)

objects_features, objects_classes = dataset_extractor.get_gt(three_classes=True)

# Directory to save the model
MODEL_DIR = DATA_PATH / "training" / "models" / "3class_ensemble"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = str(MODEL_DIR / "logs")
MODEL_PATH = str(MODEL_DIR)


# Train model parameters
TIMESTEPS_CNT = objects_features.shape[1]
FEATURES_CNT = objects_features.shape[2]
EPOCHS = 150
CELLS = 5
BATCH_SIZE = 256


# Define the model
def get_model():
    # create the model
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(TIMESTEPS_CNT, FEATURES_CNT)))
    model.add(LSTM(CELLS, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="Adam",
        metrics=["accuracy", Precision(), Recall()],
    )

    return model


classif_ensemble = Ensemble_ovo_ova()

# Load data into the model
print(50 * "-", "Loading data", 50 * "-")
classif_ensemble.load_data(
    objects_features,
    objects_classes,
    verbose=True,
    split_idx=DATASET_SPLIT_IDX,
    scaler_path=SCALER_PATH,
)

# Train the model
print(50 * "-", "Training models", 50 * "-")
classif_ensemble.train(
    get_model_func=get_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    save_path=MODEL_PATH,
    log_dir=LOG_DIR,
)

# Evaluate the performance on test data
print(50 * "-", "Evaluation on the test data", 50 * "-")
classif_ensemble.metrics()
