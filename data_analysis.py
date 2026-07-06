
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parent

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


from data_preprocessing import load_raw_data, validate_columns, inspect_dataset
from feature_engineering import (
    create_analysis_dataset,
    ZONE_1_COLUMN,
    ZONE_2_COLUMN,
    ZONE_3_COLUMN,
    TOTAL_CONSUMPTION_COLUMN,
)



sns.set_theme(style="whitegrid")

DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to outputs/data_visualizations/ and return its path."""
    # Save the chart image as a PNG file in the outputs folder.
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path




def plot_consumption_over_time(df: pd.DataFrame) -> plt.Figure:
    """
    Line plot of total electricity consumption across the full
    dataset period. Resampled to daily mean, since plotting all
    52,416 raw 10-minute points is unreadable.
    """

    # Average total power per day (too many raw points to plot directly).
    daily = (
        df.set_index("DateTime")[TOTAL_CONSUMPTION_COLUMN]
        .resample("D")
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(daily.index, daily.values, color="#1f77b4", linewidth=1.2)

    ax.set_title("Total Electricity Consumption Over Time (Daily Average)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Power Consumption")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    _save_fig(fig, "01_consumption_over_time.png")

    return fig


# Write a one-line text summary of the consumption-over-time chart.
def finding_consumption_over_time(df: pd.DataFrame) -> str:
    daily = (
        df.set_index("DateTime")[TOTAL_CONSUMPTION_COLUMN]
        .resample("D")
        .mean()
    )
    peak_day = daily.idxmax()
    low_day = daily.idxmin()

    return (
        f"Daily average demand peaked on {peak_day.date()} "
        f"({daily.max():,.0f}) and was lowest on {low_day.date()} "
        f"({daily.min():,.0f}), consistent with seasonal temperature-driven "
        f"demand across the year."
    )



def plot_average_hourly_consumption(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of average total consumption by hour of day."""

    # Average power use for each hour of the day (0-23).
    hourly_avg = (
        df.groupby("hour")[TOTAL_CONSUMPTION_COLUMN]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=hourly_avg.index,
        y=hourly_avg.values,
        color="#ff7f0e",
        ax=ax,
    )

    ax.set_title("Average Electricity Consumption by Hour of Day")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Average Total Power Consumption")

    fig.tight_layout()
    _save_fig(fig, "02_average_hourly_consumption.png")

    return fig


# Write a one-line text summary of the hourly-average chart.
def finding_average_hourly_consumption(df: pd.DataFrame) -> str:
    hourly_avg = df.groupby("hour")[TOTAL_CONSUMPTION_COLUMN].mean()
    peak_hour = hourly_avg.idxmax()
    low_hour = hourly_avg.idxmin()

    return (
        f"Demand peaks at {peak_hour}:00 ({hourly_avg.max():,.0f} average) "
        f"and is lowest at {low_hour}:00 ({hourly_avg.min():,.0f} average), "
        f"reflecting typical daily activity patterns."
    )




def plot_average_daily_consumption(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of average total consumption by day of week."""

    # Average power use for each day of the week, in Monday-Sunday order.
    daily_avg = (
        df.groupby("day_name")[TOTAL_CONSUMPTION_COLUMN]
        .mean()
        .reindex(DAY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=daily_avg.index,
        y=daily_avg.values,
        color="#2ca02c",
        ax=ax,
    )

    ax.set_title("Average Electricity Consumption by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Total Power Consumption")

    fig.tight_layout()
    _save_fig(fig, "03_average_daily_consumption.png")

    return fig


# Write a one-line text summary of the daily-average chart.
def finding_average_daily_consumption(df: pd.DataFrame) -> str:
    daily_avg = df.groupby("day_name")[TOTAL_CONSUMPTION_COLUMN].mean().reindex(DAY_ORDER)
    peak_day = daily_avg.idxmax()
    low_day = daily_avg.idxmin()
    weekend_avg = df[df["is_weekend"] == 1][TOTAL_CONSUMPTION_COLUMN].mean()
    weekday_avg = df[df["is_weekend"] == 0][TOTAL_CONSUMPTION_COLUMN].mean()

    return (
        f"{peak_day} has the highest average consumption ({daily_avg.max():,.0f}) "
        f"and {low_day} the lowest ({daily_avg.min():,.0f}). "
        f"Weekday average is {weekday_avg:,.0f} vs weekend average {weekend_avg:,.0f}."
    )




def plot_zone_comparison(df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing average consumption across the three zones."""

    # Average power use for each of the 3 zones.
    zone_cols = [ZONE_1_COLUMN, ZONE_2_COLUMN, ZONE_3_COLUMN]
    zone_labels = ["Zone 1", "Zone 2", "Zone 3"]
    zone_avg = df[zone_cols].mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=zone_labels,
        y=zone_avg.values,
        hue=zone_labels,
        legend=False,
        palette="viridis",
        ax=ax,
    )

    ax.set_title("Average Electricity Consumption by Zone")
    ax.set_xlabel("Zone")
    ax.set_ylabel("Average Power Consumption")

    fig.tight_layout()
    _save_fig(fig, "04_zone_comparison.png")

    return fig


# Write a one-line text summary of the zone-comparison chart.
def finding_zone_comparison(df: pd.DataFrame) -> str:
    zone_cols = [ZONE_1_COLUMN, ZONE_2_COLUMN, ZONE_3_COLUMN]
    zone_avg = df[zone_cols].mean()
    top_zone = zone_avg.idxmax()
    bottom_zone = zone_avg.idxmin()

    return (
        f"{top_zone} has the highest average consumption ({zone_avg.max():,.0f}), "
        f"while {bottom_zone} has the lowest ({zone_avg.min():,.0f})."
    )




def plot_temperature_vs_consumption(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of temperature against total consumption."""

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        df["Temperature"],
        df[TOTAL_CONSUMPTION_COLUMN],
        s=4,
        alpha=0.15,
        color="#d62728",
    )

    ax.set_title("Temperature vs Total Electricity Consumption")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Total Power Consumption")

    fig.tight_layout()
    _save_fig(fig, "05_temperature_vs_consumption.png")

    return fig


# Write a one-line text summary of how temperature relates to power use.
def finding_temperature_vs_consumption(df: pd.DataFrame) -> str:
    # -1 to 1: how strongly temperature and power use move together.
    correlation = df["Temperature"].corr(df[TOTAL_CONSUMPTION_COLUMN])
    direction = "positive" if correlation > 0 else "negative"

    return (
        f"Temperature and total consumption have a {direction} correlation "
        f"of {correlation:.2f}, indicating that as temperature rises, "
        f"{'consumption tends to rise too' if correlation > 0 else 'consumption tends to fall'} "
        f"(likely due to cooling/heating demand)."
    )




def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of correlations among weather variables and consumption."""

    columns = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "general diffuse flows",
        "diffuse flows",
        ZONE_1_COLUMN,
        ZONE_2_COLUMN,
        ZONE_3_COLUMN,
        TOTAL_CONSUMPTION_COLUMN,
    ]

    # How strongly each pair of columns moves together.
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )

    ax.set_title("Correlation Heatmap: Weather Variables and Consumption")

    fig.tight_layout()
    _save_fig(fig, "06_correlation_heatmap.png")

    return fig


# Find which weather variable has the strongest link to total power use.
def finding_correlation_heatmap(df: pd.DataFrame) -> str:
    columns = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "general diffuse flows",
        "diffuse flows",
    ]
    corr_with_total = (
        df[columns + [TOTAL_CONSUMPTION_COLUMN]]
        .corr()[TOTAL_CONSUMPTION_COLUMN]
        .drop(TOTAL_CONSUMPTION_COLUMN)
    )
    # Pick the variable with the biggest correlation, ignoring the sign.
    strongest = corr_with_total.abs().idxmax()

    return (
        f"Among the weather variables, {strongest} shows the strongest "
        f"relationship with total consumption (correlation = "
        f"{corr_with_total[strongest]:.2f})."
    )



def plot_weekday_vs_weekend(df: pd.DataFrame) -> plt.Figure:
    """Bonus chart: average consumption, weekday vs weekend."""

    # Average power use, grouped by weekday (0) vs weekend (1).
    grouped = (
        df.groupby("is_weekend")[TOTAL_CONSUMPTION_COLUMN]
        .mean()
        .rename({0: "Weekday", 1: "Weekend"})
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.barplot(
        x=grouped.index,
        y=grouped.values,
        hue=grouped.index,
        legend=False,
        palette="pastel",
        ax=ax,
    )

    ax.set_title("Average Consumption: Weekday vs Weekend")
    ax.set_ylabel("Average Total Power Consumption")

    fig.tight_layout()
    _save_fig(fig, "07_weekday_vs_weekend.png")

    return fig



def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a dictionary of key structural facts about the dataset,
    reusable by app.py for the Dataset Explorer page.
    """

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "start_date": df["DateTime"].min(),
        "end_date": df["DateTime"].max(),
    }




def main() -> None:
    print("=" * 80)
    print("MEMBER 2: DATASET ANALYSIS AND VISUALIZATION")
    print("=" * 80)

    # Load the raw data and check it looks right.
    raw_df = load_raw_data()
    validate_columns(raw_df)

    print("\nRunning full dataset inspection...\n")
    inspect_dataset(raw_df)

    # Build the analysis-ready dataset (adds total power + time columns).
    df = create_analysis_dataset(raw_df)

    print("\n" + "=" * 80)
    print("GENERATING MANDATORY DATASET VISUALIZATIONS")
    print("=" * 80)

    charts = [
        ("Consumption Over Time", plot_consumption_over_time, finding_consumption_over_time),
        ("Average Hourly Consumption", plot_average_hourly_consumption, finding_average_hourly_consumption),
        ("Average Daily Consumption", plot_average_daily_consumption, finding_average_daily_consumption),
        ("Zone Comparison", plot_zone_comparison, finding_zone_comparison),
        ("Temperature vs Consumption", plot_temperature_vs_consumption, finding_temperature_vs_consumption),
        ("Correlation Heatmap", plot_correlation_heatmap, finding_correlation_heatmap),
    ]

    # Make every chart, save it as a PNG, then print its text finding.
    for title, plot_fn, finding_fn in charts:
        fig = plot_fn(df)
        plt.close(fig)
        print(f"\n[{title}]")
        print(finding_fn(df))

    # Bonus chart: weekday vs weekend comparison.
    fig = plot_weekday_vs_weekend(df)
    plt.close(fig)

    print("\n" + "=" * 80)
    print(f"All charts saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()