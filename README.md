# Urban Electricity Consumption Forecasting Using Ensemble Machine Learning

## Overview

This project develops an ensemble machine learning system for forecasting total urban electricity consumption **one hour into the future**.

The system uses historical electricity demand, weather conditions, and time-based patterns to estimate future electricity consumption. Several regression models are trained and compared, including a Voting Regressor ensemble. The best-performing research model is then optimized into a smaller deployment model suitable for integration into a Streamlit dashboard.

The project addresses the real-world challenge of changing electricity demand, which makes planning and electricity distribution more difficult. Accurate short-term demand forecasting can support improved demand planning, peak-load preparation, grid management, and urban energy decision-making.

---

## Project Objective

The main objective of the project is:

> **To develop and evaluate an ensemble machine learning system for forecasting total urban electricity consumption one hour ahead.**

The project specifically aims to:

1. Prepare and analyze a real-world urban electricity consumption dataset.
2. Identify important electricity demand patterns and relationships.
3. Engineer weather, time, and historical demand features.
4. Train and compare multiple machine learning regression models.
5. Develop an ensemble model for improved forecasting performance.
6. Evaluate all models using appropriate regression metrics.
7. Optimize the final model for practical deployment.
8. Integrate the trained model into an interactive dashboard.

---

## Real-World Problem

Electricity consumption changes throughout the day and across different periods because of factors such as:

- temperature;
- humidity;
- wind speed;
- solar radiation;
- time of day;
- day of the week;
- recent electricity consumption patterns.

These variations make it difficult to estimate future demand accurately.

A forecasting system can help electricity providers and urban energy planners anticipate future demand and support:

- electricity supply planning;
- peak-demand preparation;
- grid management;
- urban energy analysis;
- operational decision-making.

---

## Dataset

The project uses the **Power Consumption of Tetouan City dataset**, a public secondary dataset containing electricity consumption and weather observations.

### Dataset Summary

| Property | Value |
|---|---|
| Number of records | 52,416 |
| Number of original columns | 9 |
| Recording interval | 10 minutes |
| Date range | 1 January 2017 to 30 December 2017 |
| Electricity zones | 3 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate timestamps | 0 |

### Original Dataset Columns

- `DateTime`
- `Temperature`
- `Humidity`
- `Wind Speed`
- `general diffuse flows`
- `diffuse flows`
- `Zone 1 Power Consumption`
- `Zone 2 Power Consumption`
- `Zone 3 Power Consumption`

---

## Forecasting Target

The three electricity consumption zones are combined to create total electricity consumption:

```text
Total Power Consumption
=
Zone 1 Power Consumption
+
Zone 2 Power Consumption
+
Zone 3 Power Consumption
```

The system forecasts electricity consumption **one hour ahead**.

Because the dataset contains one observation every 10 minutes:

```text
6 rows × 10 minutes = 60 minutes
```

Therefore, the forecasting target is created by shifting total electricity consumption six rows into the future:

```text
Target_1_Hour_Ahead
```

---

## Methodology

The project follows the pipeline below:

```text
Raw Dataset
      ↓
Data Validation
      ↓
Data Preparation
      ↓
Feature Engineering
      ↓
1-Hour Forecast Target
      ↓
Chronological Train/Test Split
      ↓
Naive Forecasting Baseline
      ↓
Individual Machine Learning Models
      ↓
Voting Regressor Ensemble
      ↓
Model Evaluation
      ↓
Deployment Model Optimization
      ↓
Saved Deployable Model
      ↓
Interactive Dashboard Integration
```

---

## Data Validation

The dataset was inspected before model development.

The following checks were performed:

- dataset file existence;
- number of rows and columns;
- expected column validation;
- missing-value detection;
- duplicate-row detection;
- DateTime parsing;
- invalid timestamp detection;
- duplicate timestamp detection;
- time-interval validation;
- numerical summary analysis.

The dataset passed the initial quality checks with:

```text
Missing values: 0
Duplicate rows: 0
Invalid DateTime values: 0
Duplicate timestamps: 0
Most common interval: 10 minutes
```

---

## Feature Engineering

Three categories of model features were created.

### Weather Features

- `Temperature`
- `Humidity`
- `Wind Speed`
- `general diffuse flows`
- `diffuse flows`

### Time Features

- `hour`
- `day_of_week`
- `month`
- `is_weekend`

### Current and Historical Demand Features

- `Total Power Consumption`
- `lag_1`
- `lag_3`
- `lag_6`
- `rolling_mean_6`

### Lag Interpretation

Because one row represents 10 minutes:

| Feature | Meaning |
|---|---|
| `lag_1` | Consumption 10 minutes ago |
| `lag_3` | Consumption 30 minutes ago |
| `lag_6` | Consumption 1 hour ago |
| `rolling_mean_6` | Average consumption over the previous hour |

The rolling average is shifted before calculation to ensure that only historical information is used and to prevent future-data leakage.

---

## Forecasting Dataset

After feature engineering:

| Dataset | Rows | Columns |
|---|---:|---:|
| Analysis dataset | 52,416 | 15 |
| Forecasting dataset | 52,404 | 20 |

Twelve rows are removed because:

```text
First 6 rows:
Insufficient historical information for lag and rolling features

Last 6 rows:
No target value available one hour into the future
```

---

## Train and Test Strategy

The data is not randomly shuffled because this is a forecasting problem.

A chronological split is used:

```text
Earlier observations
        ↓
Training data

Later observations
        ↓
Testing data
```

### Final Split

| Dataset | Records |
|---|---:|
| Training set | 41,923 |
| Testing set | 10,481 |

### Training Period

```text
2017-01-01 01:00
to
2017-10-19 04:00
```

### Testing Period

```text
2017-10-19 04:10
to
2017-12-30 22:50
```

This approach ensures that the models are evaluated on later unseen observations.

---

## Models Evaluated

The project compares the following approaches:

1. Naive Persistence Baseline
2. Linear Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor
6. Voting Regressor Ensemble

### Naive Baseline

The baseline assumes:

```text
Electricity consumption one hour from now
=
Current electricity consumption
```

The machine learning models are expected to outperform this simple forecasting approach.

---

## Ensemble Model

The Voting Regressor combines predictions from:

```text
Random Forest
+
Gradient Boosting
+
XGBoost
        ↓
Combined Ensemble Forecast
```

The final ensemble prediction is generated by combining the outputs of the three individual tree-based models.

---

## Evaluation Metrics

The models are evaluated using:

### Mean Absolute Error

```text
MAE
```

Measures the average absolute difference between actual and predicted values.

Lower values are better.

### Root Mean Squared Error

```text
RMSE
```

Measures prediction error while penalizing large mistakes more heavily.

Lower values are better.

### Coefficient of Determination

```text
R²
```

Measures how much variation in electricity consumption is explained by the model.

Values closer to `1.0` are better.

### Mean Absolute Percentage Error

```text
MAPE
```

Expresses the average prediction error as a percentage.

Lower values are better.

### Normalized RMSE

```text
NRMSE
```

Expresses RMSE relative to the average electricity demand level.

Lower values are better.

---

## Final Model Results

| Approach | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| **Voting Regressor** | **2,442.66** | **3,500.08** | **0.9385** | **3.98%** |
| XGBoost | 2,516.66 | 3,556.94 | 0.9365 | 4.15% |
| Random Forest | 2,592.13 | 3,642.31 | 0.9334 | 4.21% |
| Gradient Boosting | 2,499.69 | 3,670.70 | 0.9323 | 4.00% |
| Linear Regression | 3,160.76 | 4,656.03 | 0.8911 | 4.85% |
| Naive Baseline | 4,177.71 | 5,935.21 | 0.8231 | 6.46% |

---

## Best Research Model

The **Voting Regressor** achieved the best overall performance.

```text
MAE:   2,442.66
RMSE:  3,500.08
R²:    0.9385
MAPE:  3.98%
NRMSE: 5.40%
```

The ensemble achieved:

- the lowest MAE;
- the lowest RMSE;
- the highest R²;
- the lowest MAPE.

The results show that combining Random Forest, Gradient Boosting, and XGBoost produced better overall forecasting performance than any single model.

---

## Improvement Over the Naive Baseline

The naive baseline achieved:

```text
RMSE: 5,935.21
MAE:  4,177.71
```

The Voting Regressor achieved:

```text
RMSE: 3,500.08
MAE:  2,442.66
```

This represents approximately:

```text
41% reduction in RMSE
42% reduction in MAE
```

The results demonstrate that the machine learning approach provides substantial improvement over simply assuming that future demand will remain equal to current demand.

---

## Feature Importance

The most influential features were:

| Rank | Feature | Approximate Importance |
|---|---|---:|
| 1 | Total Power Consumption | 75.02% |
| 2 | lag_1 | 7.87% |
| 3 | general diffuse flows | 6.42% |
| 4 | hour | 4.48% |
| 5 | diffuse flows | 3.60% |

The findings indicate that:

> Current electricity demand is the strongest predictor of demand one hour ahead, while recent historical demand, solar radiation, and time of day also contribute to the forecast.

---

## Dataset Analysis Findings

The exploratory analysis identified several demand patterns.

### Consumption Over Time

The highest daily average demand occurred on:

```text
25 July 2017
Average demand: approximately 98,049
```

The lowest daily average occurred on:

```text
1 December 2017
Average demand: approximately 58,574
```

This shows substantial variation in electricity demand across the year.

### Hourly Demand Pattern

Peak average electricity consumption occurs at:

```text
20:00
Average demand: approximately 98,037
```

The lowest average electricity consumption occurs at:

```text
06:00
Average demand: approximately 50,190
```

### Day-of-Week Pattern

```text
Highest average:
Thursday → approximately 72,531

Lowest average:
Sunday → approximately 67,714
```

### Weekday vs Weekend

```text
Weekday average:
approximately 71,999

Weekend average:
approximately 69,282
```

### Zone Comparison

```text
Highest average consumption:
Zone 1 → approximately 32,345

Lowest average consumption:
Zone 3 → approximately 17,835
```

### Temperature Relationship

Temperature has a positive correlation of approximately:

```text
0.49
```

with total electricity consumption.

This indicates that higher temperatures tend to be associated with higher electricity consumption in this dataset.

---

## Mandatory Dataset Visualizations

The project generates the following visualizations:

1. Electricity consumption over time
2. Average hourly electricity consumption
3. Average electricity consumption by day of week
4. Zone consumption comparison
5. Temperature versus total electricity consumption
6. Correlation heatmap
7. Weekday versus weekend consumption

Generated visualizations are stored in:

```text
outputs/data_visualizations/
```

---

## Model Result Outputs

The model pipeline generates:

```text
outputs/model_results/
├── model_metrics.csv
├── test_predictions.csv
├── feature_importance.csv
├── best_model_summary.csv
├── deployment_model_comparison.csv
└── deployment_model_summary.csv
```

### `model_metrics.csv`

Contains performance metrics for:

- Naive Baseline
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Voting Regressor

### `test_predictions.csv`

Contains:

- DateTime
- Actual value
- Predicted value
- Prediction error
- Absolute error

### `feature_importance.csv`

Contains ranked model input features.

### `best_model_summary.csv`

Contains the final research model and performance metrics.

### `deployment_model_comparison.csv`

Contains the comparison of compact deployment candidates.

### `deployment_model_summary.csv`

Contains details of the selected lightweight deployment model.

---

## Deployment Optimization

The original Voting Regressor model was approximately:

```text
729 MB
```

This was too large for convenient application deployment.

Three smaller ensemble configurations were evaluated:

| Model | RMSE | R² | MAPE | Size |
|---|---:|---:|---:|---:|
| Compact Ensemble A | 3,534.52 | 0.9373 | 3.95% | 39.66 MB |
| **Compact Ensemble B** | **3,571.21** | **0.9360** | **3.98%** | **14.63 MB** |
| Compact Ensemble C | 3,605.80 | 0.9347 | 3.99% | 5.90 MB |

The deployment rule selected:

> The smallest model whose RMSE remained within 3% of the full research ensemble.

### Selected Deployment Model

```text
Compact Ensemble B
```

Performance:

```text
MAE:             2,468.82
RMSE:            3,571.21
R²:              0.9360
MAPE:            3.98%
NRMSE:           5.51%
Model size:      14.63 MB
RMSE degradation: 2.03%
```

The model size was reduced from:

```text
729 MB
↓
14.63 MB
```

while maintaining very similar forecasting performance.

---

## Live Prediction Pipeline

The deployment model is stored at:

```text
models/deployable_model.joblib
```

The prediction logic is implemented in:

```text
src/prediction.py
```

The pipeline:

1. loads the trained model;
2. validates the prediction inputs;
3. arranges features in the exact training order;
4. predicts total electricity consumption one hour ahead;
5. classifies the predicted demand level.

### Example Prediction

```text
Current timestamp:
2017-01-07 23:40

Actual value one hour ahead:
59,578.09

Predicted value one hour ahead:
60,789.63

Absolute error:
1,211.54

Demand level:
Moderate
```

---

## Demand-Level Classification

Forecasts are classified into understandable demand levels:

```text
Low
Moderate
High
Very High
```

The classification thresholds are based on the observed total electricity consumption distribution.

---

## Project Structure

```text
UrbanElectricityConsumptionForecasting/
│
├── app.py
├── data_analysis.py
├── train_model.py
├── optimize_model.py
├── smoke_test.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Tetuan City power consumption.csv
│
├── models/
│   └── deployable_model.joblib
│
├── outputs/
│   ├── data_visualizations/
│   │   ├── 01_consumption_over_time.png
│   │   ├── 02_average_hourly_consumption.png
│   │   ├── 03_average_daily_consumption.png
│   │   ├── 04_zone_comparison.png
│   │   ├── 05_temperature_vs_consumption.png
│   │   ├── 06_correlation_heatmap.png
│   │   └── 07_weekday_vs_weekend.png
│   │
│   └── model_results/
│       ├── model_metrics.csv
│       ├── test_predictions.csv
│       ├── feature_importance.csv
│       ├── best_model_summary.csv
│       ├── deployment_model_comparison.csv
│       └── deployment_model_summary.csv
│
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── feature_engineering.py
    └── prediction.py
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ianwambaire/UrbanElectricityConsumptionForecasting.git
```

Enter the project directory:

```bash
cd UrbanElectricityConsumptionForecasting
```

### 2. Create a Virtual Environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run Dataset Analysis

```bash
python data_analysis.py
```

This:

- validates the dataset;
- performs exploratory analysis;
- regenerates the seven mandatory visualizations.

### Train and Evaluate the Full Models

```bash
python train_model.py
```

This:

- creates the one-hour forecasting target;
- performs the chronological split;
- evaluates the naive baseline;
- trains all individual models;
- trains the Voting Regressor;
- generates final model evaluation outputs.

### Optimize the Deployment Model

```bash
python optimize_model.py
```

This:

- trains compact ensemble candidates;
- compares model size and forecasting performance;
- selects the best deployment model.

### Run the Full Project Smoke Test

```bash
python smoke_test.py
```

The smoke test verifies:

```text
1. Raw dataset exists
2. Raw dataset loads correctly
3. Analysis dataset can be created
4. Forecasting dataset can be created
5. Model result files exist
6. Visualization files exist
7. Deployable model exists
8. Deployable model loads
9. Live prediction pipeline works
```

Expected result:

```text
Total checks: 9
Passed: 9
Failed: 0

ALL PROJECT CHECKS PASSED
The project is ready for integration testing.
```

---

## Streamlit Dashboard

The final system is designed to provide:

### Overview

- project problem;
- project objective;
- dataset summary;
- final model summary;
- forecast horizon.

### Dataset Explorer

- dataset preview;
- data shape;
- missing values;
- date range;
- summary statistics.

### Consumption Analysis

- consumption over time;
- hourly demand patterns;
- day-of-week patterns;
- zone comparison;
- temperature relationship;
- correlation analysis;
- weekday versus weekend patterns.

### Model Results

- model performance comparison;
- MAE;
- RMSE;
- R²;
- MAPE;
- actual versus predicted values;
- feature importance;
- prediction error analysis.

### Forecast Consumption

- model-based electricity forecast;
- one-hour-ahead prediction;
- demand-level classification;
- deployment model information.

Run the dashboard with:

```bash
streamlit run app.py
```

---

## Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Models

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Voting Regressor

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### Deployment

- Streamlit

### Version Control

- Git
- GitHub

---

## Team Responsibilities

### Ian Wambaire — Project Lead and Machine Learning

Responsibilities:

- project setup;
- repository management;
- data validation;
- preprocessing;
- feature engineering;
- forecasting target design;
- chronological train/test split;
- baseline implementation;
- machine learning model training;
- ensemble development;
- model evaluation;
- model optimization;
- deployment model creation;
- prediction pipeline;
- system integration;
- final project validation.

### Member 2 — Dataset Analysis and Visualization

Responsibilities:

- exploratory dataset analysis;
- mandatory data visualizations;
- pattern identification;
- interpretation of electricity consumption trends.

### Member 3 — Dashboard Development and Integration

Responsibilities:

- Streamlit interface;
- dashboard navigation;
- integration of dataset visualizations;
- integration of model evaluation outputs;
- live forecast interface;
- final dashboard presentation.

---

## Current Project Status

```text
Dataset validation                 Complete
Feature engineering                Complete
1-hour forecasting pipeline        Complete
Individual model training          Complete
Naive baseline                     Complete
Voting Regressor ensemble          Complete
Model evaluation                   Complete
Deployment optimization            Complete
Live prediction pipeline           Complete
Dataset analysis                   Complete
Mandatory visualizations           Complete
Project smoke test                 Complete
Streamlit dashboard integration    In progress
```

---

## Authors

- Ian Wambaire
- Sharon Kanyi
- Rurigi Maina

Strathmore University  
Nairobi, Kenya