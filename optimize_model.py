"""
Lightweight deployment model optimization for the Urban
Electricity Consumption Forecasting project.

This script:

1. Reuses the existing one-hour forecasting dataset.
2. Preserves the official evaluation results.
3. Tests several smaller Voting Regressor configurations.
4. Measures:
   - MAE
   - RMSE
   - R²
   - MAPE
   - Normalized RMSE
   - Training time
   - Serialized model size
5. Selects the smallest model whose RMSE is within 3 percent
   of the official ensemble model.
6. Saves the selected deployable model.

The official full ensemble remains the research benchmark.

The lightweight ensemble is intended for deployment.
"""

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

from xgboost import XGBRegressor

from src.data_preprocessing import (
    load_raw_data,
)

from src.feature_engineering import (
    create_forecasting_dataset,
)

from train_model import (
    FEATURE_COLUMNS,
    FORECAST_STEPS,
    RANDOM_STATE,
    TEST_SIZE,
    calculate_regression_metrics,
    chronological_train_test_split,
    validate_chronological_split,
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

DEPLOYABLE_MODEL_PATH = (
    MODELS_DIR
    / "deployable_model.joblib"
)

COMPARISON_PATH = (
    RESULTS_DIR
    / "deployment_model_comparison.csv"
)

DEPLOYMENT_SUMMARY_PATH = (
    RESULTS_DIR
    / "deployment_model_summary.csv"
)


# ============================================================
# Official benchmark
# ============================================================

OFFICIAL_MODEL_NAME = (
    "Full Voting Regressor"
)

OFFICIAL_RMSE = (
    3500.078444904937
)

MAX_RMSE_DEGRADATION = (
    0.03
)

MAX_ACCEPTABLE_RMSE = (
    OFFICIAL_RMSE
    * (
        1
        + MAX_RMSE_DEGRADATION
    )
)


# ============================================================
# Compact model constructors
# ============================================================

def create_compact_ensemble_a() -> VotingRegressor:
    """
    Conservative compact ensemble.

    Intended to retain most of the performance of the
    original full ensemble.
    """

    random_forest = (
        RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    gradient_boosting = (
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )
    )

    xgboost = (
        XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    return VotingRegressor(
        estimators=[
            (
                "random_forest",
                random_forest,
            ),
            (
                "gradient_boosting",
                gradient_boosting,
            ),
            (
                "xgboost",
                xgboost,
            ),
        ],
        n_jobs=-1,
    )


def create_compact_ensemble_b() -> VotingRegressor:
    """
    Balanced compact ensemble.

    Designed to provide a strong compromise between
    predictive accuracy and serialized model size.
    """

    random_forest = (
        RandomForestRegressor(
            n_estimators=60,
            max_depth=16,
            min_samples_split=6,
            min_samples_leaf=2,
            max_features=0.85,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    gradient_boosting = (
        GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=3,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )
    )

    xgboost = (
        XGBRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    return VotingRegressor(
        estimators=[
            (
                "random_forest",
                random_forest,
            ),
            (
                "gradient_boosting",
                gradient_boosting,
            ),
            (
                "xgboost",
                xgboost,
            ),
        ],
        n_jobs=-1,
    )


def create_compact_ensemble_c() -> VotingRegressor:
    """
    Aggressively compressed ensemble.

    Designed to minimize deployment size while remaining
    a true ensemble machine-learning model.
    """

    random_forest = (
        RandomForestRegressor(
            n_estimators=40,
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    gradient_boosting = (
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.07,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        )
    )

    xgboost = (
        XGBRegressor(
            n_estimators=150,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    return VotingRegressor(
        estimators=[
            (
                "random_forest",
                random_forest,
            ),
            (
                "gradient_boosting",
                gradient_boosting,
            ),
            (
                "xgboost",
                xgboost,
            ),
        ],
        n_jobs=-1,
    )


# ============================================================
# Candidate models
# ============================================================

def create_candidate_models() -> dict:
    """
    Create all lightweight ensemble candidates.
    """

    return {
        "Compact Ensemble A": (
            create_compact_ensemble_a()
        ),

        "Compact Ensemble B": (
            create_compact_ensemble_b()
        ),

        "Compact Ensemble C": (
            create_compact_ensemble_c()
        ),
    }


# ============================================================
# File-size helpers
# ============================================================

def bytes_to_megabytes(
    size_bytes: int,
) -> float:
    """
    Convert bytes to megabytes.
    """

    return (
        size_bytes
        / (
            1024
            ** 2
        )
    )


def save_and_measure_model(
    model: object,
    model_name: str,
) -> tuple[
    Path,
    float,
]:
    """
    Save a candidate model temporarily using joblib
    compression and return its file size.
    """

    safe_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    candidate_path = (
        MODELS_DIR
        / f"{safe_name}.joblib"
    )

    joblib.dump(
        model,
        candidate_path,
        compress=3,
    )

    size_bytes = (
        candidate_path
        .stat()
        .st_size
    )

    size_mb = (
        bytes_to_megabytes(
            size_bytes
        )
    )

    return (
        candidate_path,
        size_mb,
    )


# ============================================================
# Train and evaluate one candidate
# ============================================================

def evaluate_candidate(
    model_name: str,
    model: VotingRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[
    VotingRegressor,
    dict,
    Path,
]:
    """
    Train, evaluate and measure one deployment candidate.
    """

    print(
        "\n" + "=" * 80
    )

    print(
        model_name.upper()
    )

    print(
        "=" * 80
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

    (
        candidate_path,
        model_size_mb,
    ) = save_and_measure_model(
        model,
        model_name,
    )

    metrics[
        "Training_Time_Seconds"
    ] = float(
        training_time
    )

    metrics[
        "Model_Size_MB"
    ] = float(
        model_size_mb
    )

    metrics[
        "RMSE_Change_Percent"
    ] = float(
        (
            metrics["RMSE"]
            - OFFICIAL_RMSE
        )
        / OFFICIAL_RMSE
        * 100
    )

    metrics[
        "Within_3_Percent_RMSE"
    ] = bool(
        metrics["RMSE"]
        <= MAX_ACCEPTABLE_RMSE
    )

    print(
        f"\nMAE: "
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

    print(
        f"Compressed model size: "
        f"{model_size_mb:.2f} MB"
    )

    print(
        f"RMSE change from official model: "
        f"{metrics['RMSE_Change_Percent']:.2f}%"
    )

    print(
        "Within acceptable RMSE limit: "
        f"{metrics['Within_3_Percent_RMSE']}"
    )

    return (
        model,
        metrics,
        candidate_path,
    )


# ============================================================
# Select deployment model
# ============================================================

def select_deployment_model(
    trained_candidates: dict,
    results: dict,
    candidate_paths: dict,
) -> tuple[
    str,
    VotingRegressor,
    dict,
]:
    """
    Select the smallest candidate whose RMSE is within
    3 percent of the official full ensemble.

    If no candidate satisfies the threshold, select the
    candidate with the lowest RMSE.
    """

    acceptable_models = [
        model_name
        for model_name, metrics
        in results.items()
        if metrics[
            "Within_3_Percent_RMSE"
        ]
    ]

    if acceptable_models:

        selected_name = min(
            acceptable_models,
            key=lambda name: (
                results[
                    name
                ][
                    "Model_Size_MB"
                ]
            ),
        )

        selection_reason = (
            "Smallest model within 3% "
            "of official ensemble RMSE"
        )

    else:

        selected_name = min(
            results,
            key=lambda name: (
                results[
                    name
                ][
                    "RMSE"
                ]
            ),
        )

        selection_reason = (
            "Lowest RMSE because no model "
            "met the 3% threshold"
        )

    selected_model = (
        trained_candidates[
            selected_name
        ]
    )

    selected_metrics = (
        results[
            selected_name
        ]
        .copy()
    )

    selected_metrics[
        "Selection_Reason"
    ] = (
        selection_reason
    )

    selected_candidate_path = (
        candidate_paths[
            selected_name
        ]
    )

    if DEPLOYABLE_MODEL_PATH.exists():
        DEPLOYABLE_MODEL_PATH.unlink()

    selected_candidate_path.replace(
        DEPLOYABLE_MODEL_PATH
    )

    return (
        selected_name,
        selected_model,
        selected_metrics,
    )


# ============================================================
# Remove unused candidate files
# ============================================================

def remove_unused_candidate_files() -> None:
    """
    Remove temporary candidate models after the final
    deployment model has been selected.
    """

    for candidate_path in (
        MODELS_DIR.glob(
            "compact_ensemble_*.joblib"
        )
    ):

        if candidate_path.exists():
            candidate_path.unlink()


# ============================================================
# Save comparison outputs
# ============================================================

def save_comparison_results(
    results: dict,
) -> pd.DataFrame:
    """
    Save all deployment candidate results.
    """

    comparison_df = (
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
            by=[
                "Within_3_Percent_RMSE",
                "Model_Size_MB",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    return comparison_df


def save_deployment_summary(
    selected_name: str,
    selected_metrics: dict,
) -> pd.DataFrame:
    """
    Save a compact summary of the selected deployment model.
    """

    summary_df = pd.DataFrame(
        [
            {
                "Deployment_Model": (
                    selected_name
                ),
                "Forecast_Horizon": (
                    "1 hour ahead"
                ),
                "Number_of_Features": (
                    len(
                        FEATURE_COLUMNS
                    )
                ),
                "MAE": (
                    selected_metrics[
                        "MAE"
                    ]
                ),
                "RMSE": (
                    selected_metrics[
                        "RMSE"
                    ]
                ),
                "R2": (
                    selected_metrics[
                        "R2"
                    ]
                ),
                "MAPE_Percent": (
                    selected_metrics[
                        "MAPE_Percent"
                    ]
                ),
                "NRMSE_Percent": (
                    selected_metrics[
                        "NRMSE_Percent"
                    ]
                ),
                "Model_Size_MB": (
                    selected_metrics[
                        "Model_Size_MB"
                    ]
                ),
                "RMSE_Change_Percent": (
                    selected_metrics[
                        "RMSE_Change_Percent"
                    ]
                ),
                "Selection_Reason": (
                    selected_metrics[
                        "Selection_Reason"
                    ]
                ),
            }
        ]
    )

    summary_df.to_csv(
        DEPLOYMENT_SUMMARY_PATH,
        index=False,
    )

    return summary_df


# ============================================================
# Main optimization pipeline
# ============================================================

def main() -> None:
    """
    Run the complete lightweight model optimization process.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "DEPLOYMENT MODEL OPTIMIZATION"
    )

    print(
        "=" * 80
    )

    print(
        "\nOfficial ensemble RMSE:"
    )

    print(
        f"{OFFICIAL_RMSE:,.2f}"
    )

    print(
        "\nMaximum acceptable RMSE:"
    )

    print(
        f"{MAX_ACCEPTABLE_RMSE:,.2f}"
    )

    print(
        "\nAllowed RMSE degradation:"
    )

    print(
        f"{MAX_RMSE_DEGRADATION * 100:.0f}%"
    )

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

    candidate_models = (
        create_candidate_models()
    )

    trained_candidates = {}

    results = {}

    candidate_paths = {}

    for (
        model_name,
        model,
    ) in candidate_models.items():

        (
            trained_model,
            metrics,
            candidate_path,
        ) = evaluate_candidate(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        trained_candidates[
            model_name
        ] = trained_model

        results[
            model_name
        ] = metrics

        candidate_paths[
            model_name
        ] = candidate_path

    comparison_df = (
        save_comparison_results(
            results
        )
    )

    (
        selected_name,
        selected_model,
        selected_metrics,
    ) = select_deployment_model(
        trained_candidates,
        results,
        candidate_paths,
    )

    remove_unused_candidate_files()

    summary_df = (
        save_deployment_summary(
            selected_name,
            selected_metrics,
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "DEPLOYMENT MODEL COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        comparison_df.round(4)
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "SELECTED DEPLOYMENT MODEL"
    )

    print(
        "=" * 80
    )

    print(
        f"\nModel: "
        f"{selected_name}"
    )

    print(
        f"RMSE: "
        f"{selected_metrics['RMSE']:,.2f}"
    )

    print(
        f"R²: "
        f"{selected_metrics['R2']:.4f}"
    )

    print(
        f"MAPE: "
        f"{selected_metrics['MAPE_Percent']:.2f}%"
    )

    print(
        f"Model size: "
        f"{selected_metrics['Model_Size_MB']:.2f} MB"
    )

    print(
        f"RMSE change: "
        f"{selected_metrics['RMSE_Change_Percent']:.2f}%"
    )

    print(
        "\nSelection reason:"
    )

    print(
        selected_metrics[
            "Selection_Reason"
        ]
    )

    print(
        "\nDeployable model saved to:"
    )

    print(
        DEPLOYABLE_MODEL_PATH
    )

    print(
        "\nComparison saved to:"
    )

    print(
        COMPARISON_PATH
    )

    print(
        "\nSummary saved to:"
    )

    print(
        DEPLOYMENT_SUMMARY_PATH
    )


if __name__ == "__main__":
    main()