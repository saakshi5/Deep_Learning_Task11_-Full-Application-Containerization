
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "api" / "model" / "incident_model.keras"
LABELS_PATH = BASE_DIR / "api" / "model" / "labels.json"


def test_model_files_exist():
    assert MODEL_PATH.is_file()
    assert LABELS_PATH.is_file()


def test_model_prediction_shape_and_probabilities():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with open(LABELS_PATH, encoding="utf-8") as file:
        labels = json.load(file)

    predictions = model.predict(
        np.asarray(
            ["wifi connection is not working"],
            dtype=str,
        ),
        verbose=0,
    )

    assert predictions.shape == (1, len(labels))
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)
    assert np.isclose(predictions.sum(axis=1)[0], 1.0, atol=1e-5)
