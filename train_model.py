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
   - MAPE
   - Normalized RMSE
8. Selects the best model using RMSE.
9. Saves:
   - final trained model
   - model metrics
   - test predictions
   - ensemble feature importance
   - best model summary

The dataset interval is 10 minutes.

Therefore:

    6 future rows = 1 hour ahead
"""

from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import joblib
import numpy as np
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
    mean_absolute_percentage_error,
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
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "best_model.joblib"
)

METRICS_PATH = (
    RESULTS_DIR
    / "model_metrics.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "test_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = (
    RESULTS_DIR
    / "feature_importance.csv"
)

BEST_MODEL_SUMMARY_PATH = (
    RESULTS_DIR
    / "best_model_summary.csv"
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
# Create output directories
# ============================================================

def create_output_directories() -> None:
    """
    Create directories required for model artifacts and
    evaluation outputs.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    The Voting Regressor combines:

    - Random Forest
    - Gradient Boosting
    - XGBoost
    """

    return VotingRegressor(
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


# ============================================================
# Calculate regression metrics
# ============================================================

def calculate_regression_metrics(
    y_true: pd.Series,
    predictions,
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Metrics:
    - MAE
    - RMSE
    - R²
    - MAPE
    - Normalized RMSE
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

    mape = (
        mean_absolute_percentage_error(
            y_true,
            predictions,
        )
        * 100
    )

    mean_actual = float(
        np.mean(y_true)
    )

    nrmse_percent = (
        rmse
        / mean_actual
        * 100
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_Percent": float(mape),
        "NRMSE_Percent": float(
            nrmse_percent
        ),
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

    The baseline predicts that electricity consumption
    one hour ahead will equal current consumption.
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
        "Training_Time_Seconds"
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
    np.ndarray,
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
        "Training_Time_Seconds"
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
        f"MAPE: "
        f"{metrics['MAPE_Percent']:.2f}%"
    )

    print(
        f"Normalized RMSE: "
        f"{metrics['NRMSE_Percent']:.2f}%"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    return (
        model,
        metrics,
        predictions,
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
            _,
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
    np.ndarray,
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

    return train_and_evaluate_model(
        "Voting Regressor",
        ensemble,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Ensemble feature importance
# ============================================================

def calculate_ensemble_feature_importance(
    ensemble: VotingRegressor,
) -> pd.DataFrame:
    """
    Calculate average feature importance across the three
    fitted tree-based ensemble components.

    Components:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    """

    importance_arrays = []

    for estimator in ensemble.estimators_:

        if hasattr(
            estimator,
            "feature_importances_",
        ):
            importance_arrays.append(
                estimator.feature_importances_
            )

    if not importance_arrays:
        raise ValueError(
            "No feature importance values "
            "were available."
        )

    average_importance = (
        np.mean(
            importance_arrays,
            axis=0,
        )
    )

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": average_importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# Save model metrics
# ============================================================

def save_model_metrics(
    results: Dict[
        str,
        Dict[str, float],
    ],
) -> pd.DataFrame:
    """
    Save metrics for all approaches.
    """

    metrics_df = (
        pd.DataFrame(
            results
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Model"
            }
        )
        .sort_values(
            by="RMSE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    return metrics_df


# ============================================================
# Save test predictions
# ============================================================

def save_test_predictions(
    test_metadata: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Save actual and predicted test values.
    """

    predictions_df = pd.DataFrame(
        {
            "DateTime": (
                test_metadata[
                    "DateTime"
                ]
                .reset_index(drop=True)
            ),

            "Actual": (
                y_test
                .reset_index(drop=True)
            ),

            "Predicted": predictions,
        }
    )

    predictions_df[
        "Error"
    ] = (
        predictions_df[
            "Predicted"
        ]
        - predictions_df[
            "Actual"
        ]
    )

    predictions_df[
        "Absolute_Error"
    ] = (
        predictions_df[
            "Error"
        ]
        .abs()
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    return predictions_df


# ============================================================
# Save best model summary
# ============================================================

def save_best_model_summary(
    model_name: str,
    metrics: Dict[str, float],
) -> pd.DataFrame:
    """
    Save a compact summary of the winning model.
    """

    summary_df = pd.DataFrame(
        [
            {
                "Best_Model": model_name,
                "Forecast_Horizon": (
                    "1 hour ahead"
                ),
                "Forecast_Steps": (
                    FORECAST_STEPS
                ),
                "Number_of_Features": (
                    len(FEATURE_COLUMNS)
                ),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE_Percent": (
                    metrics[
                        "MAPE_Percent"
                    ]
                ),
                "NRMSE_Percent": (
                    metrics[
                        "NRMSE_Percent"
                    ]
                ),
            }
        ]
    )

    summary_df.to_csv(
        BEST_MODEL_SUMMARY_PATH,
        index=False,
    )

    return summary_df


# ============================================================
# Save final artifacts
# ============================================================

def save_final_artifacts(
    ensemble: VotingRegressor,
    ensemble_metrics: Dict[str, float],
    results: Dict[
        str,
        Dict[str, float],
    ],
    test_metadata: pd.DataFrame,
    y_test: pd.Series,
    ensemble_predictions: np.ndarray,
) -> None:
    """
    Save the winning model and all dashboard-ready outputs.
    """

    create_output_directories()

    joblib.dump(
        ensemble,
        BEST_MODEL_PATH,
    )

    metrics_df = (
        save_model_metrics(
            results
        )
    )

    predictions_df = (
        save_test_predictions(
            test_metadata,
            y_test,
            ensemble_predictions,
        )
    )

    feature_importance_df = (
        calculate_ensemble_feature_importance(
            ensemble
        )
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    summary_df = (
        save_best_model_summary(
            "Voting Regressor",
            ensemble_metrics,
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL ARTIFACTS SAVED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nModel:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        f"\nMetrics:"
    )

    print(
        METRICS_PATH
    )

    print(
        f"\nPredictions:"
    )

    print(
        PREDICTIONS_PATH
    )

    print(
        f"\nFeature importance:"
    )

    print(
        FEATURE_IMPORTANCE_PATH
    )

    print(
        f"\nBest model summary:"
    )

    print(
        BEST_MODEL_SUMMARY_PATH
    )

    print(
        "\nSaved rows:"
    )

    print(
        f"Metrics: "
        f"{len(metrics_df):,}"
    )

    print(
        f"Predictions: "
        f"{len(predictions_df):,}"
    )

    print(
        f"Feature importance: "
        f"{len(feature_importance_df):,}"
    )

    print(
        f"Summary: "
        f"{len(summary_df):,}"
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
    Print the final comparison of all approaches.
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
        results_df.round(4)
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

    return results_df


# ============================================================
# Main training pipeline
# ============================================================

def main() -> None:
    """
    Run the complete forecasting, training, evaluation,
    and artifact-saving pipeline.
    """

    create_output_directories()

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
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
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

    print(
        f"MAPE: "
        f"{naive_metrics['MAPE_Percent']:.2f}%"
    )

    print(
        f"Normalized RMSE: "
        f"{naive_metrics['NRMSE_Percent']:.2f}%"
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
        ensemble_predictions,
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

    save_final_artifacts(
        trained_ensemble,
        ensemble_metrics,
        results,
        test_metadata,
        y_test,
        ensemble_predictions,
    )


if __name__ == "__main__":
    main()