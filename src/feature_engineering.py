"""
Feature engineering utilities for the Urban Electricity
Consumption Forecasting project.

This module creates:

1. Analysis features for exploratory data analysis and
   dataset visualizations.

2. Forecasting features for machine learning models.
"""

import pandas as pd


# ============================================================
# Column constants
# ============================================================

ZONE_1_COLUMN = "Zone 1 Power Consumption"

ZONE_2_COLUMN = "Zone 2  Power Consumption"

ZONE_3_COLUMN = "Zone 3  Power Consumption"

TOTAL_CONSUMPTION_COLUMN = (
    "Total Power Consumption"
)

FORECAST_TARGET_COLUMN = (
    "Target_10_Minutes_Ahead"
)


# ============================================================
# DateTime preparation
# ============================================================

def ensure_datetime(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure that the DateTime column uses pandas datetime format.

    Parameters
    ----------
    df : pd.DataFrame
        Electricity consumption dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with a parsed DateTime column.
    """

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(
        df["DateTime"]
    ):
        df["DateTime"] = pd.to_datetime(
            df["DateTime"],
            format="%m/%d/%Y %H:%M",
            errors="coerce",
        )

    invalid_datetimes = int(
        df["DateTime"]
        .isna()
        .sum()
    )

    if invalid_datetimes > 0:
        raise ValueError(
            "DateTime parsing failed for "
            f"{invalid_datetimes:,} rows."
        )

    return df


# ============================================================
# Total electricity consumption
# ============================================================

def create_total_consumption(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create total urban electricity consumption by summing
    consumption from all three zones.

    Parameters
    ----------
    df : pd.DataFrame
        Electricity consumption dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing Total Power Consumption.
    """

    df = df.copy()

    required_columns = [
        ZONE_1_COLUMN,
        ZONE_2_COLUMN,
        ZONE_3_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing zone columns: "
            f"{missing_columns}"
        )

    df[TOTAL_CONSUMPTION_COLUMN] = (
        df[ZONE_1_COLUMN]
        + df[ZONE_2_COLUMN]
        + df[ZONE_3_COLUMN]
    )

    return df


# ============================================================
# Time features
# ============================================================

def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create time-based features from the DateTime column.

    Features created:
    - hour
    - day_of_week
    - day_name
    - month
    - is_weekend

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing DateTime.

    Returns
    -------
    pd.DataFrame
        Dataset containing engineered time features.
    """

    df = ensure_datetime(df)

    df = df.copy()

    df["hour"] = (
        df["DateTime"]
        .dt.hour
    )

    df["day_of_week"] = (
        df["DateTime"]
        .dt.dayofweek
    )

    df["day_name"] = (
        df["DateTime"]
        .dt.day_name()
    )

    df["month"] = (
        df["DateTime"]
        .dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    return df


# ============================================================
# Analysis dataset
# ============================================================

def create_analysis_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the dataset used for exploratory analysis and
    mandatory dataset visualizations.

    This dataset contains:
    - original weather variables
    - three zone consumption columns
    - total electricity consumption
    - time-based features

    Parameters
    ----------
    df : pd.DataFrame
        Raw electricity dataset.

    Returns
    -------
    pd.DataFrame
        Analysis-ready dataset.
    """

    df = ensure_datetime(df)

    df = (
        df.sort_values("DateTime")
        .reset_index(drop=True)
    )

    df = create_total_consumption(df)

    df = create_time_features(df)

    return df


# ============================================================
# Historical demand features
# ============================================================

def create_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical electricity consumption features.

    Because the dataset has 10-minute intervals:

    lag_1:
        Consumption 10 minutes ago.

    lag_3:
        Consumption 30 minutes ago.

    lag_6:
        Consumption 1 hour ago.

    rolling_mean_6:
        Average consumption during the previous hour.

    The rolling mean is shifted by one row so that it only
    uses past information and avoids target leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing lag and rolling features.
    """

    df = df.copy()

    if TOTAL_CONSUMPTION_COLUMN not in df.columns:
        raise ValueError(
            "Total Power Consumption must be created "
            "before lag features."
        )

    df["lag_1"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(1)
    )

    df["lag_3"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(3)
    )

    df["lag_6"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(6)
    )

    df["rolling_mean_6"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    return df


# ============================================================
# Forecast target
# ============================================================

def create_forecast_target(
    df: pd.DataFrame,
    forecast_steps: int = 1,
) -> pd.DataFrame:
    """
    Create the future electricity consumption target.

    The dataset interval is 10 minutes.

    Default:
        forecast_steps = 1

    Therefore:
        the target is electricity consumption
        10 minutes into the future.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing total electricity consumption.

    forecast_steps : int
        Number of future rows to forecast.

    Returns
    -------
    pd.DataFrame
        Dataset containing the forecasting target.
    """

    if forecast_steps < 1:
        raise ValueError(
            "forecast_steps must be at least 1."
        )

    df = df.copy()

    df[FORECAST_TARGET_COLUMN] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(-forecast_steps)
    )

    return df


# ============================================================
# Forecasting dataset
# ============================================================

def create_forecasting_dataset(
    df: pd.DataFrame,
    forecast_steps: int = 1,
) -> pd.DataFrame:
    """
    Build the complete machine-learning forecasting dataset.

    Process:
    1. Create analysis features.
    2. Create historical demand features.
    3. Create future forecasting target.
    4. Remove rows made incomplete by shifts and rolling windows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw electricity dataset.

    forecast_steps : int
        Number of 10-minute intervals into the future.

    Returns
    -------
    pd.DataFrame
        Forecasting-ready dataset.
    """

    df = create_analysis_dataset(df)

    df = create_lag_features(df)

    df = create_forecast_target(
        df,
        forecast_steps=forecast_steps,
    )

    rows_before = len(df)

    df = (
        df.dropna()
        .reset_index(drop=True)
    )

    rows_removed = (
        rows_before
        - len(df)
    )

    print(
        "Forecasting dataset created."
    )

    print(
        f"Rows before lag/target cleanup: "
        f"{rows_before:,}"
    )

    print(
        f"Rows removed: "
        f"{rows_removed:,}"
    )

    print(
        f"Final forecasting rows: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# Feature engineering summary
# ============================================================

def print_feature_summary(
    analysis_df: pd.DataFrame,
    forecasting_df: pd.DataFrame,
) -> None:
    """
    Print a summary of engineered features.
    """

    print("=" * 80)

    print(
        "FEATURE ENGINEERING SUMMARY"
    )

    print("=" * 80)

    print(
        "\nAnalysis dataset shape:"
    )

    print(
        analysis_df.shape
    )

    print(
        "\nForecasting dataset shape:"
    )

    print(
        forecasting_df.shape
    )

    print(
        "\nAnalysis features created:"
    )

    analysis_features = [
        TOTAL_CONSUMPTION_COLUMN,
        "hour",
        "day_of_week",
        "day_name",
        "month",
        "is_weekend",
    ]

    for feature in analysis_features:
        print(
            f"- {feature}"
        )

    print(
        "\nForecasting features created:"
    )

    forecasting_features = [
        "lag_1",
        "lag_3",
        "lag_6",
        "rolling_mean_6",
        FORECAST_TARGET_COLUMN,
    ]

    for feature in forecasting_features:
        print(
            f"- {feature}"
        )

    print(
        "\nTotal consumption summary:"
    )

    print(
        analysis_df[
            TOTAL_CONSUMPTION_COLUMN
        ]
        .describe()
        .round(2)
    )

    print(
        "\nForecast target summary:"
    )

    print(
        forecasting_df[
            FORECAST_TARGET_COLUMN
        ]
        .describe()
        .round(2)
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 80
    )