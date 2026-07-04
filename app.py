"""
GridCast — embeds the exact gridcast.html UI in Streamlit, fed with live data.
Put gridcast.html next to this file.
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_preprocessing import load_raw_data              # noqa: E402
from feature_engineering import (                         # noqa: E402
    TOTAL_CONSUMPTION_COLUMN,
    create_analysis_dataset,
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "model_results"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="GridCast", layout="wide")
# strip Streamlit's default padding so the embed is full-bleed
st.markdown(
    "<style>#MainMenu,header,footer{visibility:hidden}"
    ".block-container{padding:0!important;max-width:100%!important}</style>",
    unsafe_allow_html=True,
)


@st.cache_data
def get_raw():
    df = load_raw_data()
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M", errors="coerce")
    return df


@st.cache_data
def get_analysis():
    return create_analysis_dataset(load_raw_data())


@st.cache_data
def get_csv(name):
    return pd.read_csv(RESULTS_DIR / name)


def build_payload() -> dict:
    raw = get_raw()
    adf = get_analysis()
    T = TOTAL_CONSUMPTION_COLUMN

    # ---- overview ----
    best = get_csv("best_model_summary.csv").iloc[0]
    overview = {
        "records": f"{raw.shape[0] + 1:,}",
        "zones": "3",
        "models": str(get_csv("model_metrics.csv").shape[0]),
        "bestModel": str(best["Best_Model"]),
        "bestR2": f"{best['R2']:.4f}",
        "bestMAE": f"{best['MAE']:,.2f}",
        "bestRMSE": f"{best['RMSE']:,.2f}",
        "horizon": str(best["Forecast_Horizon"]),
    }

    # ---- dataset ----
    head = raw.head(3).copy()
    preview_rows = []
    for i, (_, r) in enumerate(head.iterrows()):
        preview_rows.append([
            str(i), str(r["DateTime"])[:16],
            f"{r['Temperature']:.3f}", f"{r['Humidity']:.1f}", f"{r['Wind Speed']:.3f}",
            f"{r['general diffuse flows']:.3f}", f"{r['diffuse flows']:.3f}",
            f"{r['Zone 1 Power Consumption']:.2f}",
            f"{r['Zone 2  Power Consumption']:.2f}",
            f"{r['Zone 3  Power Consumption']:.2f}",
        ])
    desc = raw.describe().T
    summary_rows = []
    for name in ["Temperature", "Humidity", "Wind Speed",
                 "general diffuse flows", "diffuse flows",
                 "Zone 1 Power Consumption", "Zone 2  Power Consumption",
                 "Zone 3  Power Consumption"]:
        d = desc.loc[name]
        summary_rows.append([
            name.replace(" Power Consumption", ""),
            f"{d['count']:.0f}", f"{d['mean']:.2f}", f"{d['std']:.2f}",
            f"{d['min']:.2f}", f"{d['25%']:.2f}", f"{d['50%']:.2f}",
            f"{d['75%']:.2f}", f"{d['max']:.2f}",
        ])
    dataset = {
        "previewCols": ["#", "DateTime", "Temp", "Humidity", "Wind", "gen diff", "diff", "Zone 1", "Zone 2", "Zone 3"],
        "previewRows": preview_rows,
        "rows": f"{raw.shape[0]:,}", "cols": str(raw.shape[1]),
        "missing": str(int(raw.isna().sum().sum())),
        "duplicates": str(int(raw.duplicated().sum())),
        "dateRange": f"{raw['DateTime'].min().date()} → {raw['DateTime'].max().date()}",
        "colDesc": [
            {"c": "DateTime", "d": "Timestamp of the reading (10-minute intervals)"},
            {"c": "Temperature", "d": "Ambient temperature (°C)"},
            {"c": "Humidity", "d": "Relative humidity (%)"},
            {"c": "Wind Speed", "d": "Wind speed"},
            {"c": "general diffuse flows", "d": "General diffuse solar radiation flow"},
            {"c": "diffuse flows", "d": "Diffuse solar radiation flow"},
            {"c": "Zone 1 Power Consumption", "d": "Power consumption, Zone 1"},
            {"c": "Zone 2 Power Consumption", "d": "Power consumption, Zone 2"},
            {"c": "Zone 3 Power Consumption", "d": "Power consumption, Zone 3"},
        ],
        "summaryCols": ["", "count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        "summaryRows": summary_rows,
    }

    # ---- analysis ----
    daily = adf.set_index("DateTime")[T].resample("D").mean().dropna()
    over_time = [[int(ts.value // 1_000_000), round(v)] for ts, v in daily.items()]  # ms epoch
    hourly = adf.groupby("hour")[T].mean().reindex(range(24)).tolist()
    dow = adf.groupby("day_name")[T].mean().reindex(DAY_ORDER).tolist()
    zones = adf[["Zone 1 Power Consumption", "Zone 2  Power Consumption",
                 "Zone 3  Power Consumption"]].mean().round().tolist()
    samp = adf.sample(min(700, len(adf)), random_state=42)
    scatter = [[round(float(t), 1), round(float(v))]
               for t, v in zip(samp["Temperature"], samp[T])]
    heat_cols = ["Temperature", "Humidity", "Wind Speed",
                 "general diffuse flows", "diffuse flows",
                 "Zone 1 Power Consumption", "Zone 2  Power Consumption",
                 "Zone 3  Power Consumption", T]
    corr = adf[heat_cols].corr().round(2)
    weekday = float(adf.loc[adf["is_weekend"] == 0, T].mean())
    weekend = float(adf.loc[adf["is_weekend"] == 1, T].mean())
    temp_corr = float(adf["Temperature"].corr(adf[T]))
    peak_hour = int(np.nanargmax(hourly))
    peak_day_idx = int(np.nanargmax(dow))
    zmax, zmin = int(np.argmax(zones)), int(np.argmin(zones))
    weather = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]
    wc = adf[weather + [T]].corr()[T].drop(T)
    strongest = wc.abs().idxmax()

    analysis = {
        "overTime": over_time, "hourly": hourly, "dow": dow, "zones": zones, "scatter": scatter,
        "corrLabels": ["Temp", "Humid", "Wind", "gen diff", "diff", "Zone1", "Zone2", "Zone3", "Total"],
        "corr": corr.values.tolist(),
        "weekday": round(weekday), "weekend": round(weekend),
        "peakHour": f"{peak_hour}:00", "peakDay": DAY_SHORT[peak_day_idx],
        "tempCorr": ("+" if temp_corr >= 0 else "") + f"{temp_corr:.2f}",
        "findings": {
            "f1": f"Daily average demand peaked on {daily.idxmax().date()} "
                  f"({daily.max():,.0f}) and was lowest on {daily.idxmin().date()} "
                  f"({daily.min():,.0f}), consistent with seasonal temperature-driven demand.",
            "f2": f"Demand peaks at {peak_hour}:00 (~{max(hourly):,.0f}) and is lowest at "
                  f"{int(np.nanargmin(hourly))}:00 (~{min(hourly):,.0f}).",
            "f3": f"{DAY_ORDER[peak_day_idx]} has the highest average consumption "
                  f"(~{max(dow):,.0f}) and {DAY_ORDER[int(np.nanargmin(dow))]} the lowest (~{min(dow):,.0f}).",
            "f4": f"Zone {zmax+1} carries the highest average load (~{zones[zmax]:,.0f}); "
                  f"Zone {zmin+1} the lowest (~{zones[zmin]:,.0f}).",
            "f5": f"Higher temperatures drive higher demand — correlation of {temp_corr:+.2f}.",
            "f6": f"{strongest} is the strongest weather predictor (corr {wc[strongest]:+.2f}); "
                  f"humidity is negatively correlated.",
            "f7": f"Weekdays average ~{weekday:,.0f} versus ~{weekend:,.0f} on weekends.",
        },
    }

    # ---- model ----
    metrics = get_csv("model_metrics.csv").sort_values("RMSE")
    metrics_rows = [[
        str(r["Model"]), f"{r['MAE']:,.2f}", f"{r['RMSE']:,.2f}", f"{r['R2']:.4f}",
        f"{r.get('MAPE_Percent', float('nan')):.2f}",
        f"{r.get('NRMSE_Percent', float('nan')):.2f}",
        f"{r.get('Training_Time_Seconds', float('nan')):.2f}",
    ] for _, r in metrics.iterrows()]

    preds = get_csv("test_predictions.csv")
    preds["DateTime"] = pd.to_datetime(preds["DateTime"])
    dp = preds.set_index("DateTime")[["Actual", "Predicted"]].resample("D").mean().dropna()
    avp_dates = [int(ts.value // 1_000_000) for ts in dp.index]
    ps = preds.sample(min(600, len(preds)), random_state=42)
    avp_scatter = [[round(float(a)), round(float(p))]
                   for a, p in zip(ps["Actual"], ps["Predicted"])]
    err = preds["Error"] if "Error" in preds else (preds["Predicted"] - preds["Actual"])
    counts, edges = np.histogram(err, bins=30)
    centers = ((edges[:-1] + edges[1:]) / 2).round().astype(int).tolist()
    feat = get_csv("feature_importance.csv").sort_values("Importance", ascending=False)
    feat_list = [[str(r["Feature"]), float(r["Importance"])] for _, r in feat.iterrows()]

    model = {
        "bestModel": str(best["Best_Model"]),
        "mae": f"{best['MAE']:,.2f}", "rmse": f"{best['RMSE']:,.2f}", "r2": f"{best['R2']:.4f}",
        "metricsCols": ["Model", "MAE", "RMSE", "R²", "MAPE %", "NRMSE %", "Train s"],
        "metricsRows": metrics_rows,
        "rmseModels": metrics["Model"].tolist(),
        "rmseVals": metrics["RMSE"].round(2).tolist(),
        "avpDates": avp_dates,
        "actual": dp["Actual"].round().tolist(),
        "predicted": dp["Predicted"].round().tolist(),
        "avpScatter": avp_scatter,
        "errCenters": centers, "errCounts": counts.tolist(),
        "feat": feat_list,
    }

    return {"overview": overview, "dataset": dataset, "analysis": analysis, "model": model}


payload = build_payload()
html = (PROJECT_ROOT / "gridcast.html").read_text(encoding="utf-8")
html = html.replace('"__PAYLOAD_JSON__"', json.dumps(payload))
components.html(html, height=3400, scrolling=True)