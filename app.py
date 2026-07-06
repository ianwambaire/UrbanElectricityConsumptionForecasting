"""
GridCast dashboard bridge.

Python prepares the real project data and saved model results, then
injects them into gridcast.html for presentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data_preprocessing import load_raw_data
from src.feature_engineering import TOTAL_CONSUMPTION_COLUMN, create_analysis_dataset

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "gridcast.html"
RESULTS_DIR = ROOT / "outputs" / "model_results"

st.set_page_config(page_title="GridCast", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {display:none!important}
      .block-container {padding:0!important;max-width:100%!important}
      iframe {width:100%!important;border:0!important}
    </style>
    """,
    unsafe_allow_html=True,
)


# @st.cache_data means Streamlit remembers the result so it doesn't
# reload/recompute this every time the page refreshes.

# Load the raw CSV and make sure dates are real dates.
@st.cache_data
def raw_data() -> pd.DataFrame:
    df = load_raw_data().copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M", errors="coerce")
    return df


# Load the raw data plus the extra analysis columns (total power, hour, etc.).
@st.cache_data
def analysis_data() -> pd.DataFrame:
    return create_analysis_dataset(load_raw_data())


# Read one of the saved results CSV files (metrics, predictions, etc.).
@st.cache_data
def result_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")
    return pd.read_csv(path)


# Turn a value into a plain float, using 0.0 for empty/missing values.
# Used everywhere below so blank data doesn't break the dashboard JSON.
def f(value) -> float:
    return 0.0 if pd.isna(value) else float(value)


def build_payload() -> dict:
    """
    Gather every number, table, and chart value the dashboard needs
    and pack it all into one big dictionary (later turned into JSON).
    """
    raw = raw_data()
    analysis = analysis_data()
    metrics = result_csv("model_metrics.csv")
    predictions = result_csv("test_predictions.csv").copy()
    importance = result_csv("feature_importance.csv").copy()
    best = result_csv("best_model_summary.csv").iloc[0]

    predictions["DateTime"] = pd.to_datetime(predictions["DateTime"])
    # Fill in error columns if the saved CSV didn't already have them.
    if "Absolute_Error" not in predictions:
        predictions["Absolute_Error"] = (predictions["Actual"] - predictions["Predicted"]).abs()
    if "Error" not in predictions:
        predictions["Error"] = predictions["Predicted"] - predictions["Actual"]

    total = TOTAL_CONSUMPTION_COLUMN

    # Overview: top-level summary numbers shown at the top of the page.
    overview = {
        "records": f"{len(raw):,}",
        "zones": "3",
        "models": str(len(metrics)),
        "bestModel": str(best["Best_Model"]),
        "bestR2": f"{best['R2']:.4f}",
        "bestMAE": f"{best['MAE']:,.2f}",
        "bestRMSE": f"{best['RMSE']:,.2f}",
        "horizon": "1 hour",
    }

    # Dataset section: a small preview table plus column descriptions.
    preview = raw.head(8)
    preview_cols = ["#"] + [str(c) for c in preview.columns]
    preview_rows = []
    for idx, row in preview.iterrows():
        values = [str(idx)]
        for col in preview.columns:
            value = row[col]
            if col == "DateTime":
                values.append(pd.Timestamp(value).strftime("%Y-%m-%d %H:%M"))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                values.append(f"{float(value):.2f}")
            else:
                values.append(str(value))
        preview_rows.append(values)

    descriptions = {
        "DateTime": "Timestamp recorded every 10 minutes",
        "Temperature": "Ambient temperature",
        "Humidity": "Relative humidity",
        "Wind Speed": "Wind speed",
        "general diffuse flows": "General diffuse solar radiation",
        "diffuse flows": "Diffuse solar radiation",
        "Zone 1 Power Consumption": "Electricity consumption in Zone 1",
        "Zone 2  Power Consumption": "Electricity consumption in Zone 2",
        "Zone 3  Power Consumption": "Electricity consumption in Zone 3",
    }

    summary = raw.select_dtypes(include="number").describe().T
    dataset = {
        "previewCols": preview_cols,
        "previewRows": preview_rows,
        "rows": f"{len(raw):,}",
        "cols": str(raw.shape[1]),
        "missing": str(int(raw.isna().sum().sum())),
        "duplicates": str(int(raw.duplicated().sum())),
        "dateRange": f"{raw['DateTime'].min().date()} → {raw['DateTime'].max().date()}",
        "colDesc": [{"c": c, "d": descriptions.get(c, "")} for c in raw.columns],
        "summaryCols": [""] + list(summary.columns),
        "summaryRows": [[str(idx)] + [f"{f(v):.2f}" for v in row] for idx, row in summary.iterrows()],
    }

    # Analysis section: numbers behind the exploration charts.
    # Average total power per day (for the time-series chart).
    daily = analysis.set_index("DateTime")[total].resample("D").mean().dropna()
    over_time = [[int(pd.Timestamp(i).timestamp() * 1000), f(v)] for i, v in daily.items()]
    # Average total power per hour of day (0-23).
    hourly = analysis.groupby("hour")[total].mean().reindex(range(24)).fillna(0)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Average total power per day of the week.
    dow = analysis.groupby("day_name")[total].mean().reindex(day_order).fillna(0)
    zones = [
        f(analysis["Zone 1 Power Consumption"].mean()),
        f(analysis["Zone 2  Power Consumption"].mean()),
        f(analysis["Zone 3  Power Consumption"].mean()),
    ]
    # Pick a random sample of rows (max 2500) so the scatter chart isn't huge.
    sample = analysis.sample(min(2500, len(analysis)), random_state=42)
    scatter = [[f(r["Temperature"]), f(r[total])] for _, r in sample.iterrows()]

    # Columns used to build the correlation heatmap.
    corr_columns = [
        "Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows",
        "Zone 1 Power Consumption", "Zone 2  Power Consumption", "Zone 3  Power Consumption", total,
    ]
    # How strongly each pair of columns moves together (-1 to 1).
    corr_df = analysis[corr_columns].corr()
    corr_labels = ["Temp", "Humid", "Wind", "gen diff", "diff", "Zone1", "Zone2", "Zone3", "Total"]
    corr = [[f(v) for v in row] for row in corr_df.values.tolist()]

    # Compare weekday vs weekend average power use.
    weekday = f(analysis.loc[analysis["is_weekend"] == 0, total].mean())
    weekend = f(analysis.loc[analysis["is_weekend"] == 1, total].mean())
    peak_hour = int(hourly.idxmax())
    low_hour = int(hourly.idxmin())
    peak_day = str(dow.idxmax())
    low_day = str(dow.idxmin())
    temp_corr = f(analysis["Temperature"].corr(analysis[total]))

    analysis_payload = {
        "overTime": over_time,
        "hourly": [f(v) for v in hourly.tolist()],
        "dow": [f(v) for v in dow.tolist()],
        "zones": zones,
        "scatter": scatter,
        "corrLabels": corr_labels,
        "corr": corr,
        "weekday": weekday,
        "weekend": weekend,
        "peakHour": f"{peak_hour:02d}:00",
        "peakDay": peak_day[:3],
        "tempCorr": f"{temp_corr:+.2f}",
        "findings": {
            "f1": "Electricity demand varies substantially across the year.",
            "f2": f"Average demand peaks at {peak_hour:02d}:00 and is lowest at {low_hour:02d}:00.",
            "f3": f"{peak_day} has the highest average consumption and {low_day} the lowest.",
            "f4": "Zone 1 has the highest average electricity consumption.",
            "f5": f"Temperature has a correlation of {temp_corr:.2f} with total consumption, indicating an association rather than causation.",
            "f6": "The heatmap shows relationships between weather variables, zone demand, and total consumption.",
            "f7": f"Weekday demand averages about {weekday:,.0f}, compared with {weekend:,.0f} on weekends.",
        },
    }

    # Model results section: table comparing all trained models.
    wanted = [c for c in ["Model", "MAE", "RMSE", "R2", "MAPE_Percent", "NRMSE_Percent", "Training_Time_Seconds"] if c in metrics.columns]
    display_names = {"R2": "R²", "MAPE_Percent": "MAPE %", "NRMSE_Percent": "NRMSE %", "Training_Time_Seconds": "Train s"}
    sorted_metrics = metrics.sort_values("RMSE")
    metric_rows = []
    for _, row in sorted_metrics.iterrows():
        out = []
        for col in wanted:
            if col == "Model":
                out.append(str(row[col]))
            elif col == "R2":
                out.append(f"{f(row[col]):.4f}")
            else:
                out.append(f"{f(row[col]):.2f}")
        metric_rows.append(out)

    # Sort features by how much they matter to the model, biggest first.
    imp = importance.sort_values("Importance", ascending=False)
    model_payload = {
        "bestModel": str(best["Best_Model"]),
        "mae": f"{best['MAE']:,.2f}",
        "rmse": f"{best['RMSE']:,.2f}",
        "r2": f"{best['R2']:.4f}",
        "metricsCols": [display_names.get(c, c) for c in wanted],
        "metricsRows": metric_rows,
        "rmseModels": sorted_metrics["Model"].astype(str).tolist(),
        "rmseVals": [f(v) for v in sorted_metrics["RMSE"]],
        "feat": [[str(r["Feature"]), f(r["Importance"])] for _, r in imp.iterrows()],
    }

    # Prediction results section: how good the model's forecasts actually are.
    scatter_pred = predictions.sample(min(3000, len(predictions)), random_state=42)
    scatter_values = [[f(r["Actual"]), f(r["Predicted"])] for _, r in scatter_pred.iterrows()]
    # Group prediction errors into 45 buckets for a histogram chart.
    counts, edges = np.histogram(predictions["Error"].to_numpy(), bins=45)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    first_time = predictions["DateTime"].min()
    periods = {}
    # Build actual-vs-predicted lines for a few different time windows,
    # so the user can zoom into shorter periods on the chart.
    for label, days in [("Full test period", None), ("First 7 days", 7), ("First 14 days", 14), ("First 30 days", 30)]:
        if days is None:
            frame = predictions.set_index("DateTime")[["Actual", "Predicted"]].resample("D").mean().dropna().reset_index()
            note = "Daily averages are shown for the full test period to keep the chart readable."
        else:
            frame = predictions.loc[predictions["DateTime"] < first_time + pd.Timedelta(days=days), ["DateTime", "Actual", "Predicted"]].copy()
            note = "Individual 10-minute test observations are shown."
        periods[label] = {
            "dates": [int(pd.Timestamp(v).timestamp() * 1000) for v in frame["DateTime"]],
            "actual": [f(v) for v in frame["Actual"]],
            "predicted": [f(v) for v in frame["Predicted"]],
            "note": note,
        }

    # Turn a table of rows into simple text rows for display.
    def rows(frame: pd.DataFrame) -> list[list[str]]:
        return [[
            pd.Timestamp(r["DateTime"]).strftime("%Y-%m-%d %H:%M"),
            f"{f(r['Actual']):,.2f}",
            f"{f(r['Predicted']):,.2f}",
            f"{f(r['Absolute_Error']):,.2f}",
        ] for _, r in frame.iterrows()]

    prediction_payload = {
        "testRecords": f"{len(predictions):,}",
        "r2": f"{best['R2']:.4f}",
        "rmse": f"{best['RMSE']:,.2f}",
        "mape": f"{best['MAPE_Percent']:.2f}%",
        "periods": periods,
        "scatter": scatter_values,
        "errCenters": [f(v) for v in centers],
        "errCounts": [int(v) for v in counts],
        "sampleCols": ["DateTime", "Actual", "Predicted", "Absolute Error"],
        "sampleRows": rows(predictions.head(30)),
        "largestErrorRows": rows(predictions.nlargest(10, "Absolute_Error")),
        "interpretation": [
            f"The best model explains approximately {best['R2'] * 100:.1f}% of the variation in unseen electricity consumption.",
            f"The average percentage error is approximately {best['MAPE_Percent']:.2f}%.",
            "The actual and predicted lines follow similar demand patterns throughout the test period.",
            "Most scatter points lie close to the perfect-prediction diagonal.",
            "The test results show that the ensemble generalizes well to later, unseen observations.",
        ],
    }

    return {
        "overview": overview,
        "dataset": dataset,
        "analysis": analysis_payload,
        "model": model_payload,
        "prediction": prediction_payload,
    }


# Stop if the dashboard HTML file is missing.
if not HTML_PATH.exists():
    st.error(f"Missing dashboard file: {HTML_PATH}")
    st.stop()

try:
    # Read the dashboard's HTML/JS file as plain text.
    html = HTML_PATH.read_text(encoding="utf-8")
    # Build all the data and turn it into a JSON string.
    payload = json.dumps(build_payload(), ensure_ascii=False, separators=(",", ":"))
    # Swap the placeholder text in the HTML for the real data.
    html = html.replace('"__PAYLOAD_JSON__"', payload)
    # Show the finished HTML page inside the Streamlit app.
    components.html(html, height=1900, scrolling=True)
except Exception as exc:
    st.error("The dashboard could not be loaded.")
    st.exception(exc)
