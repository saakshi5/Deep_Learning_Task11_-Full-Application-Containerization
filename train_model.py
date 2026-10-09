from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "api" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Demonstration dataset. For a real submission, replace/expand these examples
# with your actual labelled incident dataset.
DATA = {
    "Network": [
        "internet connection is down",
        "cannot connect to office wifi",
        "network is very slow",
        "vpn connection keeps disconnecting",
        "server cannot be reached over network",
        "wifi signal is weak",
        "network cable is unplugged",
        "dns lookup is failing",
        "users cannot access the intranet",
        "packet loss on office network",
        "router is not responding",
        "unable to connect to remote desktop",
        "network latency is high",
        "internet access stopped working",
        "vpn login succeeds but no network access",
        "shared network drive is unavailable",
        "connection to application server times out",
        "wireless network keeps dropping",
        "ip address conflict detected",
        "cannot access website from office network",
    ],
    "Hardware": [
        "laptop will not turn on",
        "keyboard keys are not working",
        "mouse is not responding",
        "monitor screen is blank",
        "printer has a paper jam",
        "computer is overheating",
        "battery is not charging",
        "laptop fan is making noise",
        "external hard drive is not detected",
        "headset microphone is broken",
        "desktop keeps shutting down",
        "screen is flickering",
        "printer is not printing",
        "usb port is not working",
        "computer makes a beeping sound",
        "webcam is not detected",
        "power adapter is damaged",
        "keyboard is disconnected",
        "monitor displays no signal",
        "laptop hinge is damaged",
    ],
    "Software": [
        "application crashes when opened",
        "cannot install the software update",
        "program freezes during startup",
        "email application will not open",
        "spreadsheet application shows an error",
        "software license activation failed",
        "application is running very slowly",
        "unable to save the document",
        "browser closes unexpectedly",
        "software update is stuck",
        "login page is not loading in the app",
        "application displays a runtime error",
        "cannot uninstall the program",
        "database application returns an error",
        "document editor is not responding",
        "application installation failed",
        "software feature is missing",
        "file cannot be opened in the application",
        "application keeps restarting",
        "plugin is incompatible with the software",
    ],
    "Security": [
        "suspicious email asking for my password",
        "possible phishing link received",
        "account may have been compromised",
        "antivirus detected malware",
        "unauthorized login alert received",
        "computer may be infected with a virus",
        "unexpected multi factor authentication request",
        "suspicious attachment in an email",
        "password reset requested without my knowledge",
        "possible ransomware warning displayed",
        "unknown application is accessing files",
        "employee account is locked after suspicious activity",
        "confidential data may have been exposed",
        "security software has been disabled",
        "unexpected login from another country",
        "browser redirects to suspicious websites",
        "unrecognized device connected to account",
        "potential data breach reported",
        "email account is sending messages automatically",
        "suspicious activity detected on workstation",
    ],
}

LABELS = sorted(DATA.keys())
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}

texts, labels = [], []
for label, examples in DATA.items():
    for example in examples:
        texts.append(example)
        labels.append(LABEL_TO_ID[label])

# Use Python strings during splitting; avoid NumPy str128 arrays in model.fit.
x_train_text, x_test_text, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels,
)

# Explicit TensorFlow string tensors solve the Keras Invalid dtype: str128 error.
x_train = tf.constant(x_train_text, dtype=tf.string)
x_test = tf.constant(x_test_text, dtype=tf.string)
y_train = np.asarray(y_train, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode="int",
    output_sequence_length=40,
)
vectorizer.adapt(x_train)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(), dtype=tf.string, name="incident_text"),
    vectorizer,
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, mask_zero=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(len(LABELS), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=40,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1,
)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

model_path = MODEL_DIR / "incident_model.keras"
model.save(model_path)

with (MODEL_DIR / "labels.json").open("w", encoding="utf-8") as f:
    json.dump(LABELS, f, indent=2)

# Save actual measured metrics for documentation.
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "class_labels": LABELS,
    "training_examples": len(x_train_text),
    "test_examples": len(x_test_text),
}
with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

sample_texts = [
    "My office wifi keeps disconnecting",
    "My laptop battery is not charging",
    "The application crashes when I open it",
    "I received a suspicious email asking for my password",
]
sample_tensor = tf.constant(sample_texts, dtype=tf.string)
predictions = model.predict(sample_tensor, verbose=0)

print("\nSample predictions:")
for text_value, probabilities in zip(sample_texts, predictions):
    predicted_id = int(np.argmax(probabilities))
    print(
        f"- {text_value} -> {LABELS[predicted_id]} "
        f"(confidence {float(probabilities[predicted_id]):.2%})"
    )

print(f"\nSaved model: {model_path}")
print(f"Saved labels: {MODEL_DIR / 'labels.json'}")
print(f"Saved actual metrics: {MODEL_DIR / 'metrics.json'}")
