
import os

import requests
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IT Incident Classifier",
    page_icon="🖥️",
    layout="centered",
)

API_URL = os.getenv(
    "API_URL",
    "http://localhost:5000",
).rstrip("/")


# --------------------------------------------------
# 2. Application heading
# --------------------------------------------------

st.title("🖥️ IT Incident Classification")
st.write(
    "Enter an IT incident description to predict "
    "its most likely category."
)

st.caption(
    "Powered by a trained neural network and Flask REST API."
)


# --------------------------------------------------
# 3. Input section
# --------------------------------------------------

with st.form("incident_form"):
    incident_text = st.text_area(
        "Incident description",
        placeholder=(
            "Example: Unable to connect to the company network"
        ),
        max_chars=2000,
        height=140,
    )

    submitted = st.form_submit_button(
        "Classify Incident",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 4. Call the Flask API
# --------------------------------------------------

if submitted:
    if not incident_text.strip():
        st.warning("Please enter an incident description.")

    else:
        try:
            with st.spinner("Analyzing the incident..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": incident_text.strip()},
                    timeout=30,
                )

            if response.ok:
                result = response.json()

                st.success("Prediction completed.")

                st.subheader("Prediction Result")

                st.metric(
                    label="Predicted Category",
                    value=result["predicted_category"],
                )

                st.metric(
                    label="Model Confidence",
                    value=f'{result["confidence"]:.1%}',
                )

                probabilities = result.get("probabilities", {})

                if probabilities:
                    st.subheader("Class Probabilities")

                    st.bar_chart(probabilities)

                with st.expander("View API response"):
                    st.json(result)

            else:
                try:
                    error_data = response.json()
                    message = error_data.get(
                        "error",
                        "The API returned an error.",
                    )
                except ValueError:
                    message = "The API returned an invalid response."

                st.error(
                    f"Prediction failed "
                    f"(HTTP {response.status_code}): {message}"
                )

        except requests.exceptions.Timeout:
            st.error(
                "The API request timed out. Please try again."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the Flask API. "
                "Check whether the API container is running."
            )

        except (requests.exceptions.RequestException, ValueError):
            st.error(
                "An unexpected error occurred while "
                "communicating with the prediction API."
            )


# --------------------------------------------------
# 5. Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Task 11 | Dockerized Deep Learning Application"
)
