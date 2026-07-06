"""
Prediction utilities for the Urban Electricity Consumption
Forecasting project.

This module:

1. Loads the compact deployable ensemble model.
2. Validates prediction inputs.
3. Arranges features in the exact order used during training.
4. Generates a 1-hour-ahead electricity consumption forecast.
5. Assigns an understandable demand level.

The deployed model is Compact Ensemble B.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DEPLOYABLE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "deployable_model.joblib"
)


# ============================================================
# Model configuration
# ============================================================

FORECAST_HORIZON = (
    "1 hour ahead"
)

DEPLOYMENT_MODEL_NAME = (
    "Compact Ensemble B"
)


# ============================================================
# Exact feature order used during model training
# ============================================================

FEATURE_COLUMNS = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "Total Power Consumption",
    "lag_1",
    "lag_3",
    "lag_6",
    "rolling_mean_6",
]


# ============================================================
# Load deployable model
# ============================================================

@lru_cache(maxsize=1)
def load_deployable_model():
    """
    Load and cache the compact deployment model.
    """

    # Stop if the saved model file is missing.
    if not DEPLOYABLE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Deployable model not found at: "
            f"{DEPLOYABLE_MODEL_PATH}"
        )

    # Load the trained model from disk.
    # @lru_cache means this only really runs once — after that the
    # same loaded model is reused instantly.
    model = joblib.load(
        DEPLOYABLE_MODEL_PATH
    )

    return model


# ============================================================
# Validate prediction inputs
# ============================================================

def validate_prediction_inputs(
    input_data: Dict[str, float],
) -> None:
    """
    Validate that all required forecasting features are present.
    """

    # Check that every feature the model needs is present in the input.
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            "Missing prediction features: "
            f"{missing_features}"
        )

    # Check that every value is a real, non-empty number.
    for feature in FEATURE_COLUMNS:

        value = input_data[
            feature
        ]

        if pd.isna(value):
            raise ValueError(
                f"{feature} cannot be missing."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{feature} must be numeric."
            )


# ============================================================
# Prepare prediction input
# ============================================================

def prepare_prediction_input(
    input_data: Dict[str, float],
) -> pd.DataFrame:
    """
    Convert prediction inputs into the exact feature order
    expected by the trained model.
    """

    # Make sure the input is valid first.
    validate_prediction_inputs(
        input_data
    )

    # Build a one-row table with columns in the exact same order
    # the model was trained on. Wrong order would give wrong predictions.
    prediction_df = pd.DataFrame(
        [
            {
                feature: input_data[
                    feature
                ]
                for feature in FEATURE_COLUMNS
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    return prediction_df


# ============================================================
# Demand-level classification
# ============================================================

def classify_demand_level(
    predicted_consumption: float,
) -> str:
    """
    Classify predicted electricity demand.

    Thresholds are based on the observed total-consumption
    distribution in the project dataset.
    """

    # Turn the raw number into an easy-to-read label using fixed cutoffs.
    if predicted_consumption < 56500:
        return "Low"

    if predicted_consumption < 70000:
        return "Moderate"

    if predicted_consumption < 84000:
        return "High"

    return "Very High"


# ============================================================
# Make one-hour-ahead forecast
# ============================================================

def predict_one_hour_ahead(
    input_data: Dict[str, float],
) -> Dict[str, object]:
    """
    Forecast total electricity consumption one hour ahead.
    """

    # Get the trained model (loaded once, then cached).
    model = load_deployable_model()

    # Turn the raw input dict into a model-ready table.
    prediction_df = (
        prepare_prediction_input(
            input_data
        )
    )

    # Ask the model for one number: predicted total power use.
    prediction = float(
        model.predict(
            prediction_df
        )[0]
    )

    # Turn that number into a simple label (Low/Moderate/High/Very High).
    demand_level = (
        classify_demand_level(
            prediction
        )
    )

    return {
        "predicted_consumption": (
            prediction
        ),
        "forecast_horizon": (
            FORECAST_HORIZON
        ),
        "demand_level": (
            demand_level
        ),
        "model_name": (
            DEPLOYMENT_MODEL_NAME
        ),
    }


# ============================================================
# Model information
# ============================================================

def get_model_information() -> Dict[str, object]:
    """
    Return deployment model information.
    """

    return {
        "model_name": (
            DEPLOYMENT_MODEL_NAME
        ),
        "forecast_horizon": (
            FORECAST_HORIZON
        ),
        "number_of_features": (
            len(FEATURE_COLUMNS)
        ),
        "model_path": str(
            DEPLOYABLE_MODEL_PATH
        ),
    }