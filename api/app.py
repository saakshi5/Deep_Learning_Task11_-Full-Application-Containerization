
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request


# --------------------------------------------------
# 1. Initialize Flask
# --------------------------------------------------

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "incident_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

# Maximum request body size: 1 MB
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


# --------------------------------------------------
# 2. Load the trained model and labels
# --------------------------------------------------

model = None
class_names = []

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}"
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(
            isinstance(label, str) and label.strip()
            for label in class_names
        )
    ):
        raise ValueError(
            "labels.json must contain a non-empty list of category names."
        )

    logger.info("Model loaded successfully from %s", MODEL_PATH)
    logger.info("Available incident categories: %s", class_names)
    logger.info("Model input specification: %s", model.input_shape)
    logger.info("Model output specification: %s", model.output_shape)

except Exception:
    logger.exception("Failed to initialize the incident classification model.")
    raise


# --------------------------------------------------
# 3. Home endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "IT Incident Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "home": "/",
            "health": "/health",
            "predict": "/predict",
        },
    }), 200


# --------------------------------------------------
# 4. Health endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
        }), 503

    return jsonify({
        "status": "healthy",
        "service": "incident-api",
        "model_loaded": True,
        "categories": class_names,
    }), 200


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check the request content type
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json."
        }), 415

    # Parse JSON safely
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    # Extract and validate incident description
    text = data.get("text")

    if not isinstance(text, str):
        return jsonify({
            "error": "The 'text' field must be a string."
        }), 400

    text = text.strip()

    if not text:
        return jsonify({
            "error": "Incident description cannot be empty."
        }), 400

    if len(text) > 2000:
        return jsonify({
            "error": "Incident description must be at most 2000 characters."
        }), 400

    if model is None:
        logger.error("Prediction requested, but the model is unavailable.")

        return jsonify({
            "error": "The classification model is unavailable."
        }), 503

    try:
        # IMPORTANT:
        # Pass the incident description as a TensorFlow string tensor.
        # This works with models that accept raw text through
        # a Keras TextVectorization layer.

        input_tensor = tf.constant([text], dtype=tf.string)

        # Run inference
        predictions = model.predict(
            input_tensor,
            verbose=0,
        )

        probabilities = np.asarray(predictions)

        # Validate the model output
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError(
                f"Unexpected model output shape: {probabilities.shape}"
            )

        probabilities = probabilities[0]

        if probabilities.size != len(class_names):
            raise ValueError(
                "The number of model outputs does not match "
                f"the number of labels. Outputs: {probabilities.size}; "
                f"labels: {len(class_names)}."
            )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                "The model returned non-finite prediction values."
            )

        # This implementation expects a multiclass model whose outputs
        # are probabilities, usually produced by a softmax layer.
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError(
                "Model outputs are outside the expected probability range."
            )

        # Determine the predicted category
        predicted_id = int(np.argmax(probabilities))
        predicted_category = class_names[predicted_id]
        confidence = float(probabilities[predicted_id])

        # Map each category to its probability
        category_probabilities = {
            name: round(float(probabilities[index]), 4)
            for index, name in enumerate(class_names)
        }

        logger.info(
            "Prediction completed. Category=%s, confidence=%.4f",
            predicted_category,
            confidence,
        )

        return jsonify({
            "input": text,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "probabilities": category_probabilities,
        }), 200

    except Exception:
        # Keep the full technical details in the API logs.
        logger.exception(
            "Prediction failed for incident description: %s",
            text,
        )

        # Return a useful but non-sensitive error to the client.
        return jsonify({
            "error": "Prediction could not be completed. Check API logs."
        }), 500


# --------------------------------------------------
# 6. Handle oversized requests
# --------------------------------------------------

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "error": "Request body is too large."
    }), 413


# --------------------------------------------------
# 7. Handle invalid JSON
# --------------------------------------------------

@app.errorhandler(400)
def bad_request(_error):
    return jsonify({
        "error": "Invalid request. Please send a valid JSON body."
    }), 400


# --------------------------------------------------
# 8. Local development entry point
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
