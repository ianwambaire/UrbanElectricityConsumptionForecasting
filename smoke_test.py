"""
Project smoke test for the Urban Electricity Consumption
Forecasting project.

This script checks that the main project components work
together correctly.

Checks include:

1. Raw dataset existence.
2. Dataset loading.
3. Expected dataset size.
4. Analysis dataset creation.
5. Forecasting dataset creation.
6. Required result files.
7. Required visualization files.
8. Deployable model existence.
9. Deployable model loading.
10. Live prediction pipeline.

Run with:

    python smoke_test.py
"""

from pathlib import Path
from typing import Callable

from src.data_preprocessing import (
    DATA_PATH,
    load_raw_data,
)

from src.feature_engineering import (
    FORECAST_TARGET_COLUMN,
    create_analysis_dataset,
    create_forecasting_dataset,
)

from src.prediction import (
    FEATURE_COLUMNS,
    load_deployable_model,
    predict_one_hour_ahead,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "deployable_model.joblib"
)


MODEL_RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
)


VISUALIZATIONS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "data_visualizations"
)


# ============================================================
# Expected project files
# ============================================================

REQUIRED_MODEL_RESULT_FILES = [
    "model_metrics.csv",
    "test_predictions.csv",
    "feature_importance.csv",
    "best_model_summary.csv",
    "deployment_model_comparison.csv",
    "deployment_model_summary.csv",
]


REQUIRED_VISUALIZATION_FILES = [
    "01_consumption_over_time.png",
    "02_average_hourly_consumption.png",
    "03_average_daily_consumption.png",
    "04_zone_comparison.png",
    "05_temperature_vs_consumption.png",
    "06_correlation_heatmap.png",
    "07_weekday_vs_weekend.png",
]


EXPECTED_RAW_ROWS = 52416

EXPECTED_ANALYSIS_ROWS = 52416

EXPECTED_FORECASTING_ROWS = 52404

EXPECTED_RAW_COLUMNS = 9

EXPECTED_FORECAST_HORIZON_STEPS = 6


# ============================================================
# Smoke-test state
# ============================================================

PASSED_CHECKS = []

FAILED_CHECKS = []


# ============================================================
# Helper functions
# ============================================================

def print_section(
    title: str,
) -> None:
    """
    Print a formatted test section heading.
    """

    print(
        "\n" + "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )


def record_pass(
    check_name: str,
    details: str = "",
) -> None:
    """
    Record a successful smoke-test check.
    """

    PASSED_CHECKS.append(
        check_name
    )

    print(
        f"[PASS] {check_name}"
    )

    if details:
        print(
            f"       {details}"
        )


def record_fail(
    check_name: str,
    error: Exception,
) -> None:
    """
    Record a failed smoke-test check.
    """

    FAILED_CHECKS.append(
        {
            "check": check_name,
            "error": str(error),
        }
    )

    print(
        f"[FAIL] {check_name}"
    )

    print(
        f"       {error}"
    )


def run_check(
    check_name: str,
    check_function: Callable,
) -> None:
    """
    Run one smoke-test check without stopping the full suite.
    """

    try:

        details = (
            check_function()
        )

        record_pass(
            check_name,
            details or "",
        )

    except Exception as error:

        record_fail(
            check_name,
            error,
        )


# ============================================================
# Check 1: Dataset exists
# ============================================================

def check_dataset_exists() -> str:
    """
    Confirm that the raw dataset file exists.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    size_mb = (
        DATA_PATH
        .stat()
        .st_size
        / (
            1024
            ** 2
        )
    )

    return (
        f"Dataset found "
        f"({size_mb:.2f} MB)"
    )


# ============================================================
# Check 2: Dataset loads correctly
# ============================================================

def check_dataset_loading() -> str:
    """
    Confirm that the raw dataset loads successfully.
    """

    df = (
        load_raw_data()
    )

    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(
            "Unexpected number of rows. "
            f"Expected {EXPECTED_RAW_ROWS:,}, "
            f"found {len(df):,}."
        )

    if df.shape[1] != EXPECTED_RAW_COLUMNS:
        raise ValueError(
            "Unexpected number of columns. "
            f"Expected {EXPECTED_RAW_COLUMNS}, "
            f"found {df.shape[1]}."
        )

    return (
        f"{len(df):,} rows × "
        f"{df.shape[1]} columns"
    )


# ============================================================
# Check 3: Analysis dataset
# ============================================================

def check_analysis_dataset() -> str:
    """
    Confirm that the analysis dataset can be created.
    """

    raw_df = (
        load_raw_data()
    )

    analysis_df = (
        create_analysis_dataset(
            raw_df
        )
    )

    if len(
        analysis_df
    ) != EXPECTED_ANALYSIS_ROWS:
        raise ValueError(
            "Unexpected analysis dataset size. "
            f"Expected {EXPECTED_ANALYSIS_ROWS:,}, "
            f"found {len(analysis_df):,}."
        )

    required_columns = [
        "DateTime",
        "Total Power Consumption",
        "hour",
        "day_of_week",
        "day_name",
        "month",
        "is_weekend",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in analysis_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Analysis dataset is missing columns: "
            f"{missing_columns}"
        )

    return (
        f"{len(analysis_df):,} rows "
        f"with all required analysis features"
    )


# ============================================================
# Check 4: Forecasting dataset
# ============================================================

def check_forecasting_dataset() -> str:
    """
    Confirm that the 1-hour forecasting dataset can be created.
    """

    raw_df = (
        load_raw_data()
    )

    forecasting_df = (
        create_forecasting_dataset(
            raw_df,
            forecast_steps=(
                EXPECTED_FORECAST_HORIZON_STEPS
            ),
        )
    )

    if len(
        forecasting_df
    ) != EXPECTED_FORECASTING_ROWS:
        raise ValueError(
            "Unexpected forecasting dataset size. "
            f"Expected {EXPECTED_FORECASTING_ROWS:,}, "
            f"found {len(forecasting_df):,}."
        )

    if (
        FORECAST_TARGET_COLUMN
        not in forecasting_df.columns
    ):
        raise ValueError(
            "Forecast target column is missing: "
            f"{FORECAST_TARGET_COLUMN}"
        )

    return (
        f"{len(forecasting_df):,} rows "
        f"with target {FORECAST_TARGET_COLUMN}"
    )


# ============================================================
# Check 5: Model result files
# ============================================================

def check_model_result_files() -> str:
    """
    Confirm that all required model-result CSV files exist.
    """

    missing_files = []

    for filename in (
        REQUIRED_MODEL_RESULT_FILES
    ):

        file_path = (
            MODEL_RESULTS_DIR
            / filename
        )

        if not file_path.exists():
            missing_files.append(
                filename
            )

    if missing_files:
        raise FileNotFoundError(
            "Missing model result files: "
            f"{missing_files}"
        )

    return (
        f"{len(REQUIRED_MODEL_RESULT_FILES)} "
        "model result files found"
    )


# ============================================================
# Check 6: Visualization files
# ============================================================

def check_visualization_files() -> str:
    """
    Confirm that all required visualization files exist.
    """

    missing_files = []

    for filename in (
        REQUIRED_VISUALIZATION_FILES
    ):

        file_path = (
            VISUALIZATIONS_DIR
            / filename
        )

        if not file_path.exists():
            missing_files.append(
                filename
            )

    if missing_files:
        raise FileNotFoundError(
            "Missing visualization files: "
            f"{missing_files}"
        )

    return (
        f"{len(REQUIRED_VISUALIZATION_FILES)} "
        "visualization files found"
    )


# ============================================================
# Check 7: Deployable model file
# ============================================================

def check_deployable_model_exists() -> str:
    """
    Confirm that the compact deployment model exists.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Deployable model not found at: "
            f"{MODEL_PATH}"
        )

    size_mb = (
        MODEL_PATH
        .stat()
        .st_size
        / (
            1024
            ** 2
        )
    )

    return (
        f"Deployable model found "
        f"({size_mb:.2f} MB)"
    )


# ============================================================
# Check 8: Deployable model loads
# ============================================================

def check_model_loading() -> str:
    """
    Confirm that the trained deployment model loads correctly.
    """

    model = (
        load_deployable_model()
    )

    model_type = (
        type(model)
        .__name__
    )

    return (
        f"Loaded model type: "
        f"{model_type}"
    )


# ============================================================
# Check 9: Live prediction
# ============================================================

def check_live_prediction() -> str:
    """
    Run one real forecast through the complete prediction
    pipeline.
    """

    raw_df = (
        load_raw_data()
    )

    forecasting_df = (
        create_forecasting_dataset(
            raw_df,
            forecast_steps=(
                EXPECTED_FORECAST_HORIZON_STEPS
            ),
        )
    )

    sample = (
        forecasting_df
        .iloc[1000]
    )

    input_data = {
        feature: float(
            sample[
                feature
            ]
        )
        for feature in FEATURE_COLUMNS
    }

    result = (
        predict_one_hour_ahead(
            input_data
        )
    )

    predicted_value = (
        result[
            "predicted_consumption"
        ]
    )

    actual_value = float(
        sample[
            FORECAST_TARGET_COLUMN
        ]
    )

    absolute_error = abs(
        actual_value
        - predicted_value
    )

    if predicted_value <= 0:
        raise ValueError(
            "Prediction must be greater than zero."
        )

    if (
        result[
            "forecast_horizon"
        ]
        != "1 hour ahead"
    ):
        raise ValueError(
            "Unexpected forecast horizon: "
            f"{result['forecast_horizon']}"
        )

    return (
        f"Actual: {actual_value:,.2f} | "
        f"Predicted: {predicted_value:,.2f} | "
        f"Absolute error: {absolute_error:,.2f}"
    )


# ============================================================
# Final summary
# ============================================================

def print_final_summary() -> None:
    """
    Print the final smoke-test result.
    """

    print_section(
        "SMOKE TEST SUMMARY"
    )

    total_checks = (
        len(PASSED_CHECKS)
        + len(FAILED_CHECKS)
    )

    print(
        f"\nTotal checks: "
        f"{total_checks}"
    )

    print(
        f"Passed: "
        f"{len(PASSED_CHECKS)}"
    )

    print(
        f"Failed: "
        f"{len(FAILED_CHECKS)}"
    )

    if FAILED_CHECKS:

        print(
            "\nFailed checks:"
        )

        for failure in (
            FAILED_CHECKS
        ):

            print(
                f"- "
                f"{failure['check']}: "
                f"{failure['error']}"
            )

        print(
            "\nPROJECT SMOKE TEST FAILED"
        )

        raise SystemExit(1)

    print(
        "\nALL PROJECT CHECKS PASSED"
    )

    print(
        "The project is ready for "
        "integration testing."
    )


# ============================================================
# Main smoke-test pipeline
# ============================================================

def main() -> None:
    """
    Run the complete project smoke test.
    """

    print_section(
        "URBAN ELECTRICITY CONSUMPTION "
        "FORECASTING - SMOKE TEST"
    )

    run_check(
        "Raw dataset exists",
        check_dataset_exists,
    )

    run_check(
        "Raw dataset loads correctly",
        check_dataset_loading,
    )

    run_check(
        "Analysis dataset can be created",
        check_analysis_dataset,
    )

    run_check(
        "Forecasting dataset can be created",
        check_forecasting_dataset,
    )

    run_check(
        "Model result files exist",
        check_model_result_files,
    )

    run_check(
        "Visualization files exist",
        check_visualization_files,
    )

    run_check(
        "Deployable model exists",
        check_deployable_model_exists,
    )

    run_check(
        "Deployable model loads",
        check_model_loading,
    )

    run_check(
        "Live prediction pipeline works",
        check_live_prediction,
    )

    print_final_summary()


if __name__ == "__main__":
    main()