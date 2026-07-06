"""
Feature engineering utilities for the Urban Electricity
Consumption Forecasting project.

This module creates:

1. Analysis features for exploratory data analysis and
   dataset visualizations.

2. Forecasting features for machine-learning models.

The final forecasting objective is to predict total urban
electricity consumption 1 hour into the future.

The dataset contains one observation every 10 minutes,
therefore:

    6 rows = 1 hour
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
    "Target_1_Hour_Ahead"
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
        Dataset with a correctly parsed DateTime column.
    """

    df = df.copy()

    # Only convert the column if it is not already a real date type.
    if not pd.api.types.is_datetime64_any_dtype(
        df["DateTime"]
    ):
        df["DateTime"] = pd.to_datetime(
            df["DateTime"],
            format="%m/%d/%Y %H:%M",
            errors="coerce",
        )

    # Count any dates that failed to convert.
    invalid_datetimes = int(
        df["DateTime"]
        .isna()
        .sum()
    )

    # Stop if any dates are broken — we cannot build features on bad dates.
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
    Create total electricity consumption by summing
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

    # Check all 3 zone columns exist before adding them together.
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

    # Add up power use from all 3 zones to get the city total.
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
    Create time-based features from DateTime.

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

    # Pull the hour of the day (0-23) out of the date.
    df["hour"] = (
        df["DateTime"]
        .dt.hour
    )

    # Pull the day of the week as a number (0=Monday ... 6=Sunday).
    df["day_of_week"] = (
        df["DateTime"]
        .dt.dayofweek
    )

    # Pull the day of the week as a name (e.g. "Monday").
    df["day_name"] = (
        df["DateTime"]
        .dt.day_name()
    )

    # Pull the month number (1-12) out of the date.
    df["month"] = (
        df["DateTime"]
        .dt.month
    )

    # Mark 1 if the day is Saturday or Sunday, else 0.
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

    The analysis dataset contains:
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

    # Put rows in time order — needed for lag/rolling features later.
    df = (
        df.sort_values("DateTime")
        .reset_index(drop=True)
    )

    # Add total power column.
    df = create_total_consumption(df)

    # Add hour/day/month/weekend columns.
    df = create_time_features(df)

    return df


# ============================================================
# Historical electricity demand features
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
        Average electricity consumption during the
        previous hour.

    The rolling mean is shifted by one row before calculating
    the average. This ensures that only historical information
    is used and prevents data leakage.

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

    # Consumption 10 minutes ago (shift row down by 1).
    df["lag_1"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(1)
    )

    # Consumption 30 minutes ago (shift row down by 3).
    df["lag_3"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(3)
    )

    # Consumption 1 hour ago (shift row down by 6).
    df["lag_6"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(6)
    )

    # Average consumption over the last hour.
    # Shifted by 1 first so today's value is never included (no cheating).
    df["rolling_mean_6"] = (
        df[TOTAL_CONSUMPTION_COLUMN]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    return df


# ============================================================
# One-hour forecasting target
# ============================================================

def create_forecast_target(
    df: pd.DataFrame,
    forecast_steps: int = 6,
) -> pd.DataFrame:
    """
    Create the future electricity consumption target.

    The dataset interval is 10 minutes.

    Default:
        forecast_steps = 6

    Therefore:
        the target represents electricity consumption
        1 hour into the future.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing total electricity consumption.

    forecast_steps : int
        Number of future 10-minute intervals to forecast.

        Examples:
            1 = 10 minutes ahead
            3 = 30 minutes ahead
            6 = 1 hour ahead

    Returns
    -------
    pd.DataFrame
        Dataset containing the future forecasting target.
    """

    if forecast_steps < 1:
        raise ValueError(
            "forecast_steps must be at least 1."
        )

    df = df.copy()

    # Pull consumption from "forecast_steps" rows in the future
    # up into the current row. This becomes what we want to predict.
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
    forecast_steps: int = 6,
) -> pd.DataFrame:
    """
    Build the complete machine-learning forecasting dataset.

    Process:
    1. Create analysis features.
    2. Create historical electricity-demand features.
    3. Create the future forecasting target.
    4. Remove rows made incomplete by shifts and rolling windows.

    The default forecasting horizon is 1 hour.

    Parameters
    ----------
    df : pd.DataFrame
        Raw electricity dataset.

    forecast_steps : int
        Number of 10-minute intervals into the future.

        Default:
            6 intervals = 1 hour.

    Returns
    -------
    pd.DataFrame
        Forecasting-ready dataset.
    """

    # Step 1: add total-power and time columns.
    df = create_analysis_dataset(df)

    # Step 2: add lag / rolling-average history columns.
    df = create_lag_features(df)

    # Step 3: add the future target column we want to predict.
    df = create_forecast_target(
        df,
        forecast_steps=forecast_steps,
    )

    rows_before = len(df)

    # Step 4: drop rows with empty cells.
    # (First/last rows have no history or no future value, so they're unusable.)
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
    Print a summary of all engineered features.
    """

    print("=" * 80)

    print(
        "FEATURE ENGINEERING SUMMARY"
    )

    print("=" * 80)

    print(
        "\nForecast horizon:"
    )

    print(
        "1 hour ahead"
    )

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