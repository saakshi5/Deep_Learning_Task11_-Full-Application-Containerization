
# Task 11: Full Application Containerization

## Project Overview

This project demonstrates the containerization of a deep
learning IT incident classification application using Docker.

The application contains:
- A trained TensorFlow/Keras neural network
- A Flask REST API for prediction
- A Streamlit user interface
- Dockerfiles for the API and UI
- Docker Compose for container integration
- Automated model and API tests

## Architecture

Browser
  -> Streamlit UI (port 8501)
  -> Flask API (port 5000)
  -> Trained neural network
  -> JSON prediction response

## Prerequisites

- Python 3.11
- Docker Desktop
- Docker Compose
- Internet access for installing dependencies

## Step 1: Install training dependencies

    pip install -r requirements-training.txt

## Step 2: Train the model

    python train_model.py

The training process creates:
- api/model/incident_model.keras
- api/model/labels.json

## Step 3: Build the containers

    docker compose build

## Step 4: Start the application

    docker compose up -d

## Step 5: Verify the application

API health:
http://localhost:5000/health

Streamlit UI:
http://localhost:8501

## Step 6: Run automated model tests

    python -m pytest tests/test_model.py -v

## Step 7: Run API tests

Start the containers first, then run:

    python -m pytest tests/test_api.py -v

## Step 8: View logs

    docker compose logs -f

## Step 9: Stop the application

    docker compose down

## Prediction API

Endpoint: POST /predict

Content-Type: application/json

Example request:

    {
      "text": "Unable to connect to the company network"
    }

The response includes the predicted category, confidence,
and probabilities for the available classes.

## Important Notes

The included dataset is a small demonstration dataset.
For real-world use, train and evaluate the model on a
representative labelled dataset and validate its performance.

Keep the model assets available before building the API image.
