# Urban Electricity Consumption Forecasting

A machine learning project that predicts total urban electricity consumption **one hour ahead** using weather, time, and recent demand data.

The project uses the Tetouan City Power Consumption dataset and compares several regression models, including a Voting Regressor ensemble.

## Live Dashboard

[Open the Streamlit Dashboard](https://ianwambaire-urbanelectricityconsumptionforecasting-app-puqyec.streamlit.app/)

## Dataset

The project uses a public secondary dataset containing:

- 52,416 records
- 10-minute recording intervals
- weather variables
- electricity consumption from three zones

The three zones are combined to create total electricity consumption.

## Models

The following approaches were evaluated:

- Naive Baseline
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Voting Regressor

The data was split chronologically, with the earlier 80% used for training and the later 20% used for testing.

## Results

The **Voting Regressor** achieved the best overall performance.

| Model | RMSE | R² | MAPE |
|---|---:|---:|---:|
| Voting Regressor | 3,500.08 | 0.9385 | 3.98% |
| XGBoost | 3,556.94 | 0.9365 | 4.15% |
| Random Forest | 3,642.31 | 0.9334 | 4.21% |
| Gradient Boosting | 3,670.70 | 0.9323 | 4.00% |
| Linear Regression | 4,656.03 | 0.8911 | 4.85% |
| Naive Baseline | 5,935.21 | 0.8231 | 6.46% |

The Voting Regressor reduced RMSE by approximately **41%** compared with the naive baseline.

## Dashboard

The Streamlit dashboard includes:

- project overview
- dataset exploration
- electricity consumption analysis
- model comparison
- actual vs predicted test results

Run the dashboard locally with:

```bash
streamlit run app.py
```

## Setup

```bash
git clone https://github.com/ianwambaire/UrbanElectricityConsumptionForecasting.git
cd UrbanElectricityConsumptionForecasting

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

For Windows:

```bash
.venv\Scripts\activate
```

## Technologies

Python, Pandas, Scikit-learn, XGBoost, Plotly and Streamlit.

## Team

- Ian Wambaire
- Sharon Kanyi
- Rurigi Maina
- Kyle Edwin
- DenzelSam Omondi
- Nancy Nduta
- Macklee Gitonga

**Strathmore University — Nairobi, Kenya**