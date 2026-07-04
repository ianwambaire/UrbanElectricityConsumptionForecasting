"""
Data loading and validation utilities for the Urban Electricity
Consumption Forecasting project.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "Tetuan City power consumption.csv"
)


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


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Tetouan electricity consumption dataset.

    Returns
    -------
    pd.DataFrame
        Raw electricity consumption data.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Check whether all expected columns exist in the dataset.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected columns: "
            f"{missing_columns}"
        )


def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Print a complete inspection report for the raw dataset.
    """

    print("=" * 80)
    print("URBAN ELECTRICITY CONSUMPTION DATASET INSPECTION")
    print("=" * 80)

    print("\n1. DATASET SHAPE")
    print("-" * 80)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print("-" * 80)

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(f"{index}. {column}")

    print("\n3. DATA TYPES")
    print("-" * 80)
    print(df.dtypes)

    print("\n4. MISSING VALUES")
    print("-" * 80)

    missing_values = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(missing_values)

    print(
        "\nTotal missing values:",
        f"{missing_values.sum():,}",
    )

    print("\n5. DUPLICATE ROWS")
    print("-" * 80)

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )

    print("\n6. DATETIME INSPECTION")
    print("-" * 80)

    datetime_values = pd.to_datetime(
        df["DateTime"],
        dayfirst=True,
        errors="coerce",
    )

    invalid_datetimes = int(
        datetime_values.isna().sum()
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

        sorted_datetimes = (
            datetime_values
            .sort_values()
        )

        interval_counts = (
            sorted_datetimes
            .diff()
            .value_counts()
            .head(10)
        )

        print(
            "\nMost common time intervals:"
        )

        print(interval_counts)

    print("\n7. NUMERICAL SUMMARY")
    print("-" * 80)

    print(
        df.describe()
        .T
        .round(2)
    )

    print("\n8. ZONE CONSUMPTION SUMMARY")
    print("-" * 80)

    zone_columns = [
        "Zone 1 Power Consumption",
        "Zone 2  Power Consumption",
        "Zone 3  Power Consumption",
    ]

    zone_summary = (
        df[zone_columns]
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

    print(zone_summary)

    print("\n9. INITIAL VALIDATION RESULT")
    print("-" * 80)

    if (
        missing_values.sum() == 0
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

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


def main() -> None:
    """
    Run the dataset inspection.
    """

    df = load_raw_data()

    validate_columns(df)

    inspect_dataset(df)


if __name__ == "__main__":
    main()
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

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# Validate expected columns
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
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

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

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

    df = df.copy()

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )

    return df


# ============================================================
# Inspect dataset
# ============================================================

def inspect_dataset(df: pd.DataFrame) -> None:
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

    missing_values = (
        df.isna()
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

    duplicate_rows = int(
        df.duplicated().sum()
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

    datetime_values = pd.to_datetime(
        df["DateTime"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )

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

        sorted_datetimes = (
            datetime_values
            .sort_values()
        )

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

    numerical_summary = (
        df.describe()
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

    zone_summary = (
        df[zone_columns]
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

    df = load_raw_data()

    validate_columns(df)

    inspect_dataset(df)


if __name__ == "__main__":
    main()