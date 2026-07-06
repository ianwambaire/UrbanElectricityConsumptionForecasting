"""
Data loading and validation utilities for the Urban Electricity
Consumption Forecasting project.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "Tetuan City power consumption.csv"
)


# ============================================================
# Expected dataset columns
# ============================================================

EXPECTED_COLUMNS = [
    "DateTime",
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
]


# ============================================================
# Load dataset
# ============================================================

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Tetouan electricity consumption dataset.

    Returns
    -------
    pd.DataFrame
        Raw electricity consumption data.
    """

    # Stop early if the CSV file is missing.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    # Read the CSV file into a table (DataFrame).
    return pd.read_csv(DATA_PATH)


# ============================================================
# Validate expected columns
# ============================================================

def validate_columns(
    df: pd.DataFrame,
) -> None:
    """
    Check whether all expected columns exist in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    # Build a list of any expected columns that are not in the data.
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    # If any columns are missing, stop and tell the user which ones.
    if missing_columns:
        raise ValueError(
            "Missing expected columns: "
            f"{missing_columns}"
        )


# ============================================================
# Parse DateTime column
# ============================================================

def parse_datetime_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the DateTime column into pandas datetime format.

    The dataset uses:
        month/day/year hour:minute

    Example:
        1/1/2017 0:00

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Copy of the dataset with a parsed DateTime column.
    """

    # Make a copy so we don't change the original data by accident.
    df = df.copy()

    # Turn the text dates into real datetime values.
    # Any date that cannot be read becomes "NaT" (empty date) instead of crashing.
    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )

    return df


# ============================================================
# Inspect dataset
# ============================================================

def inspect_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Print a complete inspection report for the dataset.

    The report includes:
    - shape
    - columns
    - data types
    - missing values
    - duplicate rows
    - datetime quality
    - date range
    - time interval
    - numerical summary
    - zone consumption summary
    """

    print("=" * 80)

    print(
        "URBAN ELECTRICITY CONSUMPTION "
        "DATASET INSPECTION"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # 1. Dataset shape
    # --------------------------------------------------------

    print("\n1. DATASET SHAPE")

    print("-" * 80)

    print(
        f"Rows: {df.shape[0]:,}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )


    # --------------------------------------------------------
    # 2. Column names
    # --------------------------------------------------------

    print("\n2. COLUMN NAMES")

    print("-" * 80)

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(
            f"{index}. {column}"
        )


    # --------------------------------------------------------
    # 3. Data types
    # --------------------------------------------------------

    print("\n3. DATA TYPES")

    print("-" * 80)

    print(
        df.dtypes
    )


    # --------------------------------------------------------
    # 4. Missing values
    # --------------------------------------------------------

    print("\n4. MISSING VALUES")

    print("-" * 80)

    # Count empty (missing) cells in each column, biggest count first.
    missing_values = (
        df
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing_values
    )

    total_missing_values = int(
        missing_values.sum()
    )

    print(
        "\nTotal missing values:",
        f"{total_missing_values:,}",
    )


    # --------------------------------------------------------
    # 5. Duplicate rows
    # --------------------------------------------------------

    print("\n5. DUPLICATE ROWS")

    print("-" * 80)

    # Count rows that are exact copies of another row.
    duplicate_rows = int(
        df
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )


    # --------------------------------------------------------
    # 6. DateTime inspection
    # --------------------------------------------------------

    print("\n6. DATETIME INSPECTION")

    print("-" * 80)

    # Try to convert the DateTime column into real dates.
    datetime_values = pd.to_datetime(
        df["DateTime"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )

    # Count how many dates failed to convert (bad/invalid dates).
    invalid_datetimes = int(
        datetime_values
        .isna()
        .sum()
    )

    print(
        f"Invalid DateTime values: "
        f"{invalid_datetimes:,}"
    )

    if invalid_datetimes == 0:

        print(
            "Start date:",
            datetime_values.min(),
        )

        print(
            "End date:",
            datetime_values.max(),
        )

        duplicate_timestamps = int(
            datetime_values
            .duplicated()
            .sum()
        )

        print(
            "Duplicate timestamps:",
            f"{duplicate_timestamps:,}",
        )

        # Put the dates in order from oldest to newest.
        sorted_datetimes = (
            datetime_values
            .sort_values()
        )

        # Work out the time gap between each row and the next,
        # then show the 10 most common gaps (e.g. "10 minutes").
        interval_counts = (
            sorted_datetimes
            .diff()
            .value_counts()
            .head(10)
        )

        print(
            "\nMost common time intervals:"
        )

        print(
            interval_counts
        )


    # --------------------------------------------------------
    # 7. Numerical summary
    # --------------------------------------------------------

    print("\n7. NUMERICAL SUMMARY")

    print("-" * 80)

    # Get basic stats (mean, min, max, etc.) for every number column.
    numerical_summary = (
        df
        .describe()
        .T
        .round(2)
    )

    print(
        numerical_summary
    )


    # --------------------------------------------------------
    # 8. Zone consumption summary
    # --------------------------------------------------------

    print("\n8. ZONE CONSUMPTION SUMMARY")

    print("-" * 80)

    zone_columns = [
        "Zone 1 Power Consumption",
        "Zone 2  Power Consumption",
        "Zone 3  Power Consumption",
    ]

    # Get min/mean/median/max/std for each zone's power use.
    zone_summary = (
        df[
            zone_columns
        ]
        .agg(
            [
                "min",
                "mean",
                "median",
                "max",
                "std",
            ]
        )
        .T
        .round(2)
    )

    print(
        zone_summary
    )


    # --------------------------------------------------------
    # 9. Initial validation result
    # --------------------------------------------------------

    print("\n9. INITIAL VALIDATION RESULT")

    print("-" * 80)

    # Data is "clean" only if there are no missing values, no
    # duplicate rows, and no bad dates.
    if (
        total_missing_values == 0
        and duplicate_rows == 0
        and invalid_datetimes == 0
    ):

        print(
            "Dataset passed the initial "
            "quality checks."
        )

    else:

        print(
            "Dataset requires cleaning "
            "before modelling."
        )


    print(
        "\n" + "=" * 80
    )

    print(
        "INSPECTION COMPLETE"
    )

    print(
        "=" * 80
    )


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    """
    Run the complete dataset inspection process.
    """

    # Load the data, check it has the right columns, then print the report.
    df = load_raw_data()

    validate_columns(
        df
    )

    inspect_dataset(
        df
    )


if __name__ == "__main__":
    main()