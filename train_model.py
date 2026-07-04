"""
Machine learning training pipeline for the Urban Electricity
Consumption Forecasting project.

Major Step 3:
- Build the forecasting dataset.
- Select model features.
- Perform a chronological train/test split.
- Validate the split before model training.
"""

from typing import Tuple

import pandas as pd

from src.data_preprocessing import load_raw_data
from src.feature_engineering import (
    FORECAST_TARGET_COLUMN,
    TOTAL_CONSUMPTION_COLUMN,
    create_forecasting_dataset,
)


# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.20

FORECAST_STEPS = 1


# ============================================================
# Model features
# ============================================================

WEATHER_FEATURES = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
]

TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

DEMAND_FEATURES = [
    TOTAL_CONSUMPTION_COLUMN,
    "lag_1",
    "lag_3",
    "lag_6",
    "rolling_mean_6",
]

FEATURE_COLUMNS = (
    WEATHER_FEATURES
    + TIME_FEATURES
    + DEMAND_FEATURES
)

TARGET_COLUMN = FORECAST_TARGET_COLUMN


# ============================================================
# Validate model columns
# ============================================================

def validate_model_columns(
    df: pd.DataFrame,
) -> None:
    """
    Confirm that all required model features and the target
    are present in the forecasting dataset.
    """

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing model columns: "
            f"{missing_columns}"
        )


# ============================================================
# Prepare model data
# ============================================================

def prepare_model_data(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Select the final machine-learning features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasting-ready dataset.

    Returns
    -------
    X : pd.DataFrame
        Model input features.

    y : pd.Series
        Future electricity consumption target.
    """

    validate_model_columns(df)

    X = (
        df[FEATURE_COLUMNS]
        .copy()
    )

    y = (
        df[TARGET_COLUMN]
        .copy()
    )

    return X, y


# ============================================================
# Chronological split
# ============================================================

def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split the forecasting dataset chronologically.

    Earlier observations are used for training.
    Later observations are used for testing.

    This is more realistic than random splitting for
    a forecasting problem.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasting-ready dataset.

    test_size : float
        Proportion of observations reserved for testing.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    train_metadata
    test_metadata
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not df["DateTime"].is_monotonic_increasing:
        raise ValueError(
            "Forecasting data must be sorted "
            "chronologically before splitting."
        )

    X, y = prepare_model_data(df)

    split_index = int(
        len(df) * (1 - test_size)
    )

    X_train = (
        X.iloc[:split_index]
        .copy()
    )

    X_test = (
        X.iloc[split_index:]
        .copy()
    )

    y_train = (
        y.iloc[:split_index]
        .copy()
    )

    y_test = (
        y.iloc[split_index:]
        .copy()
    )

    train_metadata = (
        df[["DateTime"]]
        .iloc[:split_index]
        .copy()
    )

    test_metadata = (
        df[["DateTime"]]
        .iloc[split_index:]
        .copy()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_metadata,
        test_metadata,
    )


# ============================================================
# Validate chronological split
# ============================================================

def validate_chronological_split(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> None:
    """
    Verify that all training observations occur before
    all testing observations.
    """

    train_end = (
        train_metadata["DateTime"]
        .max()
    )

    test_start = (
        test_metadata["DateTime"]
        .min()
    )

    if train_end >= test_start:
        raise ValueError(
            "Chronological split validation failed. "
            "Training data overlaps testing data."
        )


# ============================================================
# Print modelling summary
# ============================================================

def print_split_summary(
    forecasting_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> None:
    """
    Print a clear summary of the forecasting dataset,
    selected features, and chronological split.
    """

    print("=" * 80)
    print(
        "FORECASTING DATASET AND "
        "CHRONOLOGICAL SPLIT"
    )
    print("=" * 80)

    print("\n1. FORECASTING CONFIGURATION")
    print("-" * 80)

    print(
        "Forecast horizon: "
        "10 minutes ahead"
    )

    print(
        f"Total forecasting rows: "
        f"{len(forecasting_df):,}"
    )

    print(
        f"Training proportion: "
        f"{(1 - TEST_SIZE) * 100:.0f}%"
    )

    print(
        f"Testing proportion: "
        f"{TEST_SIZE * 100:.0f}%"
    )

    print("\n2. MODEL FEATURES")
    print("-" * 80)

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):
        print(
            f"{index}. {feature}"
        )

    print(
        f"\nTotal model features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print("\n3. TARGET")
    print("-" * 80)

    print(
        TARGET_COLUMN
    )

    print("\n4. TRAINING DATA")
    print("-" * 80)

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        "Training start:",
        train_metadata["DateTime"].min(),
    )

    print(
        "Training end:",
        train_metadata["DateTime"].max(),
    )

    print("\n5. TESTING DATA")
    print("-" * 80)

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    print(
        "Testing start:",
        test_metadata["DateTime"].min(),
    )

    print(
        "Testing end:",
        test_metadata["DateTime"].max(),
    )

    print("\n6. TARGET SUMMARY")
    print("-" * 80)

    print("\nTraining target:")

    print(
        y_train
        .describe()
        .round(2)
    )

    print("\nTesting target:")

    print(
        y_test
        .describe()
        .round(2)
    )

    print("\n7. DATA QUALITY CHECK")
    print("-" * 80)

    training_missing = int(
        X_train.isna().sum().sum()
    )

    testing_missing = int(
        X_test.isna().sum().sum()
    )

    print(
        f"Missing values in training features: "
        f"{training_missing:,}"
    )

    print(
        f"Missing values in testing features: "
        f"{testing_missing:,}"
    )

    print(
        f"Missing values in training target: "
        f"{y_train.isna().sum():,}"
    )

    print(
        f"Missing values in testing target: "
        f"{y_test.isna().sum():,}"
    )

    print("\n8. SPLIT VALIDATION")
    print("-" * 80)

    print(
        "Chronological split passed."
    )

    print(
        "All training observations occur "
        "before all testing observations."
    )

    print("\n" + "=" * 80)
    print(
        "MODEL DATA PREPARATION COMPLETE"
    )
    print("=" * 80)


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    """
    Run forecasting data preparation and chronological split.
    """

    raw_df = load_raw_data()

    forecasting_df = (
        create_forecasting_dataset(
            raw_df,
            forecast_steps=FORECAST_STEPS,
        )
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_metadata,
        test_metadata,
    ) = chronological_train_test_split(
        forecasting_df,
        test_size=TEST_SIZE,
    )

    validate_chronological_split(
        train_metadata,
        test_metadata,
    )

    print_split_summary(
        forecasting_df,
        X_train,
        X_test,
        y_train,
        y_test,
        train_metadata,
        test_metadata,
    )


if __name__ == "__main__":
    main()