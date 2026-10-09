
import os

import requests


BASE_URL = os.getenv(
    "API_TEST_URL",
    "http://localhost:5000",
)


def test_health_endpoint():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_home_endpoint():
    response = requests.get(
        f"{BASE_URL}/",
        timeout=10,
    )

    assert response.status_code == 200
    assert "service" in response.json()


def test_valid_prediction():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "text": "Unable to connect to the company network"
        },
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_category"] in {
        "Network",
        "Hardware",
        "Software",
        "Security",
    }

    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["probabilities"], dict)


def test_empty_input():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "   "},
        timeout=10,
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_missing_text():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={},
        timeout=10,
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_non_string_text():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": 123},
        timeout=10,
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_non_json_request():
    response = requests.post(
        f"{BASE_URL}/predict",
        data="not json",
        headers={"Content-Type": "text/plain"},
        timeout=10,
    )

    assert response.status_code == 415


def test_oversized_description():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "a" * 2001},
        timeout=10,
    )

    assert response.status_code == 400
    assert "error" in response.json()
