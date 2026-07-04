"""
Machine learning training pipeline for the Urban Electricity
Consumption Forecasting project.

This pipeline:

1. Loads the electricity consumption dataset.
2. Creates a 1-hour-ahead forecasting target.
3. Performs a chronological train/test split.
4. Evaluates a naive persistence baseline.
5. Trains individual regression models.
6. Trains a Voting Regressor ensemble.
7. Evaluates all approaches using:
   - MAE
   - RMSE
   - R²
8. Selects the best approach based on RMSE.

The dataset interval is 10 minutes.

Therefore:

    6 future rows = 1 hour ahead
"""

from time import perf_counter
from typing import Dict, Tuple

import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.data_preprocessing import (
    load_raw_data,
)

from src.feature_engineering import (
    FORECAST_TARGET_COLUMN,
    TOTAL_CONSUMPTION_COLUMN,
    create_forecasting_dataset,
)


# ============================================================
# Project configuration
# ============================================================

TEST_SIZE = 0.20

FORECAST_STEPS = 6

RANDOM_STATE = 42


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


TARGET_COLUMN = (
    FORECAST_TARGET_COLUMN
)


# ============================================================
# Validate model columns
# ============================================================

def validate_model_columns(
    df: pd.DataFrame,
) -> None:
    """
    Confirm that all required model features and the target
    exist in the forecasting dataset.
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
# Chronological train/test split
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
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not df[
        "DateTime"
    ].is_monotonic_increasing:
        raise ValueError(
            "Forecasting data must be sorted "
            "chronologically before splitting."
        )

    X, y = prepare_model_data(df)

    split_index = int(
        len(df)
        * (1 - test_size)
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
        train_metadata[
            "DateTime"
        ]
        .max()
    )

    test_start = (
        test_metadata[
            "DateTime"
        ]
        .min()
    )

    if train_end >= test_start:
        raise ValueError(
            "Chronological split validation failed. "
            "Training data overlaps testing data."
        )


# ============================================================
# Reusable model constructors
# ============================================================

def create_random_forest() -> RandomForestRegressor:
    """
    Create the Random Forest model.
    """

    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def create_gradient_boosting() -> GradientBoostingRegressor:
    """
    Create the Gradient Boosting model.
    """

    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def create_xgboost() -> XGBRegressor:
    """
    Create the XGBoost model.
    """

    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ============================================================
# Individual models
# ============================================================

def create_individual_models() -> Dict[str, object]:
    """
    Create all individual regression models.
    """

    return {
        "Linear Regression": (
            LinearRegression()
        ),

        "Random Forest": (
            create_random_forest()
        ),

        "Gradient Boosting": (
            create_gradient_boosting()
        ),

        "XGBoost": (
            create_xgboost()
        ),
    }


# ============================================================
# Voting Regressor ensemble
# ============================================================

def create_voting_ensemble() -> VotingRegressor:
    """
    Create the ensemble model.

    The Voting Regressor combines predictions from:

    - Random Forest
    - Gradient Boosting
    - XGBoost

    The final forecast is the average prediction from
    the three component models.
    """

    ensemble = VotingRegressor(
        estimators=[
            (
                "random_forest",
                create_random_forest(),
            ),
            (
                "gradient_boosting",
                create_gradient_boosting(),
            ),
            (
                "xgboost",
                create_xgboost(),
            ),
        ],
        n_jobs=-1,
    )

    return ensemble


# ============================================================
# Calculate regression metrics
# ============================================================

def calculate_regression_metrics(
    y_true: pd.Series,
    predictions,
) -> Dict[str, float]:
    """
    Calculate MAE, RMSE and R².
    """

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = (
        mean_squared_error(
            y_true,
            predictions,
        )
        ** 0.5
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# Naive persistence baseline
# ============================================================

def evaluate_naive_baseline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a persistence baseline.

    The baseline assumes that electricity consumption
    one hour into the future will equal current electricity
    consumption.
    """

    predictions = (
        X_test[
            TOTAL_CONSUMPTION_COLUMN
        ]
        .to_numpy()
    )

    metrics = (
        calculate_regression_metrics(
            y_test,
            predictions,
        )
    )

    metrics[
        "Training Time Seconds"
    ] = 0.0

    return metrics


# ============================================================
# Train one model
# ============================================================

def train_and_evaluate_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    object,
    Dict[str, float],
]:
    """
    Train and evaluate one machine-learning model.
    """

    print(
        f"\nTraining {model_name}..."
    )

    start_time = (
        perf_counter()
    )

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        perf_counter()
        - start_time
    )

    predictions = (
        model.predict(
            X_test
        )
    )

    metrics = (
        calculate_regression_metrics(
            y_test,
            predictions,
        )
    )

    metrics[
        "Training Time Seconds"
    ] = float(
        training_time
    )

    print(
        f"MAE: "
        f"{metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE: "
        f"{metrics['RMSE']:,.2f}"
    )

    print(
        f"R²: "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    return (
        model,
        metrics,
    )


# ============================================================
# Train individual models
# ============================================================

def train_individual_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    Dict[str, object],
    Dict[str, Dict[str, float]],
]:
    """
    Train and evaluate all individual models.
    """

    models = (
        create_individual_models()
    )

    trained_models = {}

    results = {}

    print(
        "\n" + "=" * 80
    )

    print(
        "INDIVIDUAL MODEL TRAINING"
    )

    print(
        "=" * 80
    )

    for model_name, model in models.items():

        (
            trained_model,
            metrics,
        ) = train_and_evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        trained_models[
            model_name
        ] = trained_model

        results[
            model_name
        ] = metrics

    return (
        trained_models,
        results,
    )


# ============================================================
# Train ensemble
# ============================================================

def train_voting_ensemble(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    VotingRegressor,
    Dict[str, float],
]:
    """
    Train and evaluate the Voting Regressor ensemble.
    """

    print(
        "\n" + "=" * 80
    )

    print(
        "ENSEMBLE MODEL TRAINING"
    )

    print(
        "=" * 80
    )

    ensemble = (
        create_voting_ensemble()
    )

    (
        trained_ensemble,
        metrics,
    ) = train_and_evaluate_model(
        "Voting Regressor",
        ensemble,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    return (
        trained_ensemble,
        metrics,
    )


# ============================================================
# Print final comparison
# ============================================================

def print_final_comparison(
    results: Dict[
        str,
        Dict[str, float],
    ],
) -> pd.DataFrame:
    """
    Print the final comparison of:

    - naive baseline
    - individual models
    - ensemble model
    """

    results_df = (
        pd.DataFrame(
            results
        )
        .T
        .sort_values(
            by="RMSE",
            ascending=True,
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL MODEL AND BASELINE COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        results_df.round(
            {
                "MAE": 2,
                "RMSE": 2,
                "R2": 4,
                "Training Time Seconds": 2,
            }
        )
    )

    best_name = (
        results_df.index[0]
    )

    print(
        "\nBest approach by RMSE:"
    )

    print(
        best_name
    )

    print(
        "\nBest approach metrics:"
    )

    print(
        results_df
        .iloc[0]
        .round(4)
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL MODEL COMPARISON COMPLETE"
    )

    print(
        "=" * 80
    )

    return results_df


# ============================================================
# Print forecasting setup
# ============================================================

def print_forecasting_summary(
    forecasting_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> None:
    """
    Print forecasting setup and chronological split.
    """

    print(
        "=" * 80
    )

    print(
        "FORECASTING SETUP"
    )

    print(
        "=" * 80
    )

    print(
        "\nForecast horizon: "
        "1 hour ahead"
    )

    print(
        f"Forecast steps: "
        f"{FORECAST_STEPS}"
    )

    print(
        f"Total forecasting rows: "
        f"{len(forecasting_df):,}"
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    print(
        "\nTraining period:"
    )

    print(
        train_metadata[
            "DateTime"
        ]
        .min(),
        "→",
        train_metadata[
            "DateTime"
        ]
        .max(),
    )

    print(
        "\nTesting period:"
    )

    print(
        test_metadata[
            "DateTime"
        ]
        .min(),
        "→",
        test_metadata[
            "DateTime"
        ]
        .max(),
    )

    print(
        f"\nNumber of model features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nTarget:"
    )

    print(
        TARGET_COLUMN
    )


# ============================================================
# Main training pipeline
# ============================================================

def main() -> None:
    """
    Run the complete 1-hour-ahead ensemble forecasting pipeline.
    """

    raw_df = (
        load_raw_data()
    )

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

    print_forecasting_summary(
        forecasting_df,
        X_train,
        X_test,
        train_metadata,
        test_metadata,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "NAIVE FORECASTING BASELINE"
    )

    print(
        "=" * 80
    )

    naive_metrics = (
        evaluate_naive_baseline(
            X_test,
            y_test,
        )
    )

    print(
        f"\nMAE: "
        f"{naive_metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE: "
        f"{naive_metrics['RMSE']:,.2f}"
    )

    print(
        f"R²: "
        f"{naive_metrics['R2']:.4f}"
    )

    (
        trained_models,
        results,
    ) = train_individual_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    (
        trained_ensemble,
        ensemble_metrics,
    ) = train_voting_ensemble(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results[
        "Voting Regressor"
    ] = ensemble_metrics

    results[
        "Naive Baseline"
    ] = naive_metrics

    print_final_comparison(
        results
    )


if __name__ == "__main__":
    main()