"""
Streamlit dashboard for the Urban Electricity Consumption
Forecasting project.

Pages:
1. Overview
2. Dataset Explorer
3. Consumption Analysis
4. Model Results
5. Forecast Consumption
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_preprocessing import load_raw_data  # noqa: E402
from prediction import (  # noqa: E402
    FEATURE_COLUMNS,
    get_model_information,
    predict_one_hour_ahead,
)

VIZ_DIR = PROJECT_ROOT / "outputs" / "data_visualizations"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "model_results"

st.set_page_config(
    page_title="Urban Electricity Consumption Forecasting",
    layout="wide",
)


# ============================================================
# Cached data loaders
# ============================================================

@st.cache_data
def get_raw_data() -> pd.DataFrame:
    df = load_raw_data()
    df["DateTime"] = pd.to_datetime(
        df["DateTime"], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    return df


@st.cache_data
def get_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


# ============================================================
# Sidebar navigation
# ============================================================

PAGES = [
    "Overview",
    "Dataset Explorer",
    "Consumption Analysis",
    "Model Results",
    "Forecast Consumption",
]

page = st.sidebar.radio("Navigate", PAGES)


# ============================================================
# Page 1: Overview
# ============================================================

if page == "Overview":
    st.title("Urban Electricity Consumption Forecasting")
    st.subheader("Using Ensemble Machine Learning")

    st.markdown(
        """
        Electricity demand changes throughout the day because of
        temperature, humidity, wind speed, solar radiation, time of
        day, day of week, and recent electricity usage. Providers
        need accurate forecasts to plan supply, prepare for peak
        demand, and support grid management.

        This project forecasts **total urban electricity consumption
        one hour ahead**, using historical demand, weather conditions,
        and time-based patterns, combined through an ensemble of
        machine learning models (Voting Regressor of Random Forest,
        Gradient Boosting, and XGBoost).

        **Dataset:** Power Consumption of Tetouan City (public dataset,
        10-minute intervals, 3 urban zones).
        """
    )

    best_summary = get_csv("best_model_summary.csv").iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", "52,417")
    col2.metric("Consumption Zones", "3")
    col3.metric("Models Compared", "6")
    col4.metric("Best Model", best_summary["Best_Model"])

    col5, col6, col7 = st.columns(3)
    col5.metric("Best R²", f"{best_summary['R2']:.4f}")
    col6.metric("Best MAE", f"{best_summary['MAE']:,.2f}")
    col7.metric("Forecast Horizon", best_summary["Forecast_Horizon"])


# ============================================================
# Page 2: Dataset Explorer
# ============================================================

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")

    df = get_raw_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))

    col4, col5 = st.columns(2)
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))
    col5.metric(
        "Date Range",
        f"{df['DateTime'].min().date()} to {df['DateTime'].max().date()}",
    )

    st.subheader("Column Descriptions")
    st.table(
        pd.DataFrame(
            {
                "Column": [
                    "DateTime",
                    "Temperature",
                    "Humidity",
                    "Wind Speed",
                    "general diffuse flows",
                    "diffuse flows",
                    "Zone 1 Power Consumption",
                    "Zone 2  Power Consumption",
                    "Zone 3  Power Consumption",
                ],
                "Description": [
                    "Timestamp of the reading (10-minute intervals)",
                    "Ambient temperature (°C)",
                    "Relative humidity (%)",
                    "Wind speed",
                    "General diffuse solar radiation flow",
                    "Diffuse solar radiation flow",
                    "Power consumption, Zone 1",
                    "Power consumption, Zone 2",
                    "Power consumption, Zone 3",
                ],
            }
        )
    )

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T)


# ============================================================
# Page 3: Consumption Analysis
# ============================================================

elif page == "Consumption Analysis":
    st.title("Consumption Analysis")

    charts = [
        (
            "01_consumption_over_time.png",
            "Electricity Consumption Over Time",
            "Daily average demand peaked on 25 July 2017 (~98,049) and "
            "was lowest on 1 December 2017 (~58,574), consistent with "
            "seasonal temperature-driven demand across the year.",
        ),
        (
            "02_average_hourly_consumption.png",
            "Average Hourly Consumption",
            "Demand peaks at 20:00 (~98,037 average) and is lowest at "
            "06:00 (~50,190 average), reflecting typical daily activity "
            "patterns.",
        ),
        (
            "03_average_daily_consumption.png",
            "Average Consumption by Day of Week",
            "Thursday has the highest average consumption (~72,531) and "
            "Sunday the lowest (~67,714).",
        ),
        (
            "04_zone_comparison.png",
            "Zone Comparison",
            "Zone 1 has the highest average consumption (~32,345), while "
            "Zone 3 has the lowest (~17,835).",
        ),
        (
            "05_temperature_vs_consumption.png",
            "Temperature vs Electricity Consumption",
            "Temperature and total consumption have a positive "
            "correlation of ~0.49 — higher temperatures tend to be "
            "associated with higher electricity consumption, likely due "
            "to cooling demand.",
        ),
        (
            "06_correlation_heatmap.png",
            "Correlation Heatmap",
            "General diffuse flows shows the strongest relationship "
            "among weather variables with total consumption.",
        ),
        (
            "07_weekday_vs_weekend.png",
            "Weekday vs Weekend Consumption",
            "Weekday average consumption (~71,999) is higher than "
            "weekend average (~69,282).",
        ),
    ]

    for filename, title, finding in charts:
        st.subheader(title)
        image_path = VIZ_DIR / filename
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"Chart not found: {filename}")
        st.caption(finding)


# ============================================================
# Page 4: Model Results
# ============================================================

elif page == "Model Results":
    st.title("Model Results")

    best_summary = get_csv("best_model_summary.csv").iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", best_summary["Best_Model"])
    col2.metric("MAE", f"{best_summary['MAE']:,.2f}")
    col3.metric("RMSE", f"{best_summary['RMSE']:,.2f}")
    col4.metric("R²", f"{best_summary['R2']:.4f}")

    st.subheader("Model Comparison")
    metrics_df = get_csv("model_metrics.csv").sort_values("RMSE")
    st.dataframe(metrics_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics_df["Model"], metrics_df["RMSE"], color="#1f77b4")
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison by RMSE (lower is better)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Actual vs Predicted Consumption Over Time")
    predictions_df = get_csv("test_predictions.csv")
    predictions_df["DateTime"] = pd.to_datetime(predictions_df["DateTime"])
    hourly = (
        predictions_df.set_index("DateTime")[["Actual", "Predicted"]]
        .resample("D")
        .mean()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hourly.index, hourly["Actual"], label="Actual", linewidth=1.2)
    ax.plot(hourly.index, hourly["Predicted"], label="Predicted", linewidth=1.2)
    ax.set_ylabel("Total Power Consumption")
    ax.set_title("Actual vs Predicted (Daily Average)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Actual vs Predicted Scatter")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            predictions_df["Actual"],
            predictions_df["Predicted"],
            s=4,
            alpha=0.15,
            color="#d62728",
        )
        min_val = min(predictions_df["Actual"].min(), predictions_df["Predicted"].min())
        max_val = max(predictions_df["Actual"].max(), predictions_df["Predicted"].max())
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        fig.tight_layout()
        st.pyplot(fig)

    with col_b:
        st.subheader("Prediction Error Distribution")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(predictions_df["Error"], bins=50, color="#2ca02c")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Feature Importance")
    importance_df = get_csv("feature_importance.csv").sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#9467bd")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)


# ============================================================
# Page 5: Forecast Consumption
# ============================================================

elif page == "Forecast Consumption":
    st.title("Forecast Electricity Consumption")

    model_info = get_model_information()
    st.caption(
        f"Deployed model: {model_info['model_name']} — "
        f"forecast horizon: {model_info['forecast_horizon']}"
    )

    df = get_raw_data()

    st.markdown("Enter current conditions and recent demand:")

    col1, col2 = st.columns(2)

    with col1:
        forecast_date = st.date_input("Date", value=df["DateTime"].max().date())
        forecast_time = st.time_input("Time", value=df["DateTime"].max().time())
        temperature = st.number_input("Temperature (°C)", value=float(df["Temperature"].mean()))
        humidity = st.number_input("Humidity (%)", value=float(df["Humidity"].mean()))
        wind_speed = st.number_input("Wind Speed", value=float(df["Wind Speed"].mean()))

    with col2:
        general_diffuse_flows = st.number_input(
            "General Diffuse Flows", value=float(df["general diffuse flows"].mean())
        )
        diffuse_flows = st.number_input(
            "Diffuse Flows", value=float(df["diffuse flows"].mean())
        )
        current_consumption = st.number_input(
            "Current Total Power Consumption",
            value=60000.0,
        )
        lag_1 = st.number_input("Consumption 10 Minutes Ago (lag_1)", value=60000.0)
        lag_3 = st.number_input("Consumption 30 Minutes Ago (lag_3)", value=60000.0)

    lag_6 = st.number_input("Consumption 1 Hour Ago (lag_6)", value=60000.0)
    rolling_mean_6 = st.number_input(
        "Average Consumption, Previous Hour (rolling_mean_6)", value=60000.0
    )

    if st.button("Forecast Electricity Consumption"):
        combined_datetime = pd.Timestamp.combine(forecast_date, forecast_time)

        input_data = {
            "Temperature": temperature,
            "Humidity": humidity,
            "Wind Speed": wind_speed,
            "general diffuse flows": general_diffuse_flows,
            "diffuse flows": diffuse_flows,
            "hour": combined_datetime.hour,
            "day_of_week": combined_datetime.dayofweek,
            "month": combined_datetime.month,
            "is_weekend": int(combined_datetime.dayofweek in (5, 6)),
            "Total Power Consumption": current_consumption,
            "lag_1": lag_1,
            "lag_3": lag_3,
            "lag_6": lag_6,
            "rolling_mean_6": rolling_mean_6,
        }

        assert set(input_data.keys()) == set(FEATURE_COLUMNS)

        try:
            result = predict_one_hour_ahead(input_data)
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
        else:
            st.success("Forecast generated")
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Predicted Future Electricity Consumption",
                f"{result['predicted_consumption']:,.2f}",
            )
            col2.metric("Demand Level", result["demand_level"])
            col3.metric("Forecast Horizon", result["forecast_horizon"])
