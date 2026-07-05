"""
GridCast
Urban Electricity Consumption Forecasting Dashboard

Pages:
1. Overview
2. Dataset Explorer
3. Consumption Analysis
4. Model Results
5. Prediction Results

The dashboard presents dataset exploration, machine learning model
evaluation, and predictions generated for the unseen chronological
test set.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_preprocessing import load_raw_data
from src.feature_engineering import (
    TOTAL_CONSUMPTION_COLUMN,
    create_analysis_dataset,
)


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
)

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# ============================================================
# GridCast visual theme
# ============================================================

BACKGROUND_COLOR = "#0b0b0d"

CARD_COLOR = "#141417"

SECONDARY_CARD_COLOR = "#1a1a1f"

PRIMARY_RED = "#f9423a"

LIGHT_RED = "#ff8078"

DARK_RED = "#c9271f"

COMPARISON_BLUE = "#4aa3ff"

TEXT_COLOR = "#f4f4f6"

SECONDARY_TEXT_COLOR = "#c8c8d0"

MUTED_TEXT_COLOR = "#8f8f98"

GRID_COLOR = "rgba(255,255,255,0.08)"


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="GridCast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    f"""
    <style>

        .stApp {{
            background:
                radial-gradient(
                    circle at 85% 0%,
                    rgba(249, 66, 58, 0.07),
                    transparent 28rem
                ),
                {BACKGROUND_COLOR};

            color: {TEXT_COLOR};
        }}


        .block-container {{
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1450px;
        }}


        [data-testid="stSidebar"] {{
            background: #0d0d10;
            border-right: 1px solid rgba(255,255,255,0.08);
        }}


        [data-testid="stSidebar"] h1 {{
            color: {TEXT_COLOR};
        }}


        [data-testid="stSidebar"] p {{
            color: {MUTED_TEXT_COLOR};
        }}


        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.08);
        }}


        h1,
        h2,
        h3 {{
            color: {TEXT_COLOR};
            letter-spacing: -0.02em;
        }}


        h1 {{
            font-weight: 800;
        }}


        p {{
            color: {SECONDARY_TEXT_COLOR};
        }}


        .small-muted {{
            color: {MUTED_TEXT_COLOR};
            font-size: 0.93rem;
            margin-top: -0.5rem;
        }}


        .red-line {{
            width: 58px;
            height: 4px;
            border-radius: 999px;
            margin-bottom: 1rem;

            background:
                linear-gradient(
                    90deg,
                    {PRIMARY_RED},
                    {DARK_RED}
                );
        }}


        [data-testid="stMetric"] {{
            background: {CARD_COLOR};

            border:
                1px solid
                rgba(255,255,255,0.08);

            border-radius: 14px;

            padding: 1rem 1.1rem;
        }}


        [data-testid="stMetricLabel"] {{
            color: {MUTED_TEXT_COLOR};
        }}


        [data-testid="stMetricValue"] {{
            color: {TEXT_COLOR};
            font-weight: 700;
        }}


        button[data-baseweb="tab"] {{
            color: {MUTED_TEXT_COLOR};
        }}


        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {PRIMARY_RED};
        }}


        [data-baseweb="select"] > div {{
            background: {CARD_COLOR};
            border-color: rgba(255,255,255,0.12);
        }}


        [data-testid="stDataFrame"] {{
            border:
                1px solid
                rgba(255,255,255,0.08);

            border-radius: 12px;

            overflow: hidden;
        }}


        hr {{
            border-color: rgba(255,255,255,0.08);
        }}


        footer {{
            visibility: hidden;
        }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Cached data loaders
# ============================================================

@st.cache_data
def get_raw_data() -> pd.DataFrame:
    """
    Load the original Tetouan City dataset.
    """

    dataframe = load_raw_data()

    dataframe["DateTime"] = pd.to_datetime(
        dataframe["DateTime"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )

    return dataframe


@st.cache_data
def get_analysis_data() -> pd.DataFrame:
    """
    Create the analysis-ready dataset.
    """

    return create_analysis_dataset(
        load_raw_data()
    )


@st.cache_data
def load_result_csv(
    filename: str,
) -> pd.DataFrame:
    """
    Load a saved model-result CSV.
    """

    path = (
        RESULTS_DIR
        / filename
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Required file was not found: {path}"
        )

    return pd.read_csv(
        path
    )


# ============================================================
# Shared helper functions
# ============================================================

def page_header(
    title: str,
    subtitle: str,
) -> None:
    """
    Display a consistent page title.
    """

    st.title(
        title
    )

    st.markdown(
        f'<p class="small-muted">{subtitle}</p>',
        unsafe_allow_html=True,
    )

    st.divider()


def create_metric_row(
    metrics: list[tuple[str, str]],
) -> None:
    """
    Display a row of metric cards.
    """

    columns = st.columns(
        len(metrics)
    )

    for column, (
        label,
        value,
    ) in zip(
        columns,
        metrics,
    ):

        column.metric(
            label,
            value,
        )


def style_figure(
    figure,
    title: str | None = None,
    height: int = 430,
):
    """
    Apply the GridCast dark chart theme.
    """

    layout_settings = {
        "height": height,

        "paper_bgcolor": CARD_COLOR,

        "plot_bgcolor": CARD_COLOR,

        "font": {
            "color": TEXT_COLOR,
            "family": "Arial",
        },

        "margin": {
            "l": 35,
            "r": 25,
            "t": 55 if title else 25,
            "b": 40,
        },

        "hoverlabel": {
            "bgcolor": SECONDARY_CARD_COLOR,
            "font_color": TEXT_COLOR,
            "bordercolor": PRIMARY_RED,
        },

        "legend": {
            "bgcolor": "rgba(0,0,0,0)",

            "font": {
                "color": TEXT_COLOR,
            },
        },
    }


    if title:

        layout_settings[
            "title"
        ] = {
            "text": title,

            "font": {
                "color": TEXT_COLOR,
                "size": 18,
            },
        }


    figure.update_layout(
        **layout_settings
    )


    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        color=MUTED_TEXT_COLOR,
        linecolor=GRID_COLOR,
    )


    figure.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        color=MUTED_TEXT_COLOR,
        linecolor=GRID_COLOR,
    )


    return figure


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title(
        "⚡ GridCast"
    )

    st.caption(
        "Urban Electricity Consumption Forecasting"
    )

    st.divider()


    selected_page = st.radio(
        "Navigate",
        [
            "Overview",
            "Dataset Explorer",
            "Consumption Analysis",
            "Model Results",
            "Prediction Results",
        ],
        label_visibility="collapsed",
    )


    st.divider()


    st.caption(
        "Prediction task"
    )

    st.markdown(
        "**Total electricity demand**"
    )


    st.caption(
        "Forecast horizon"
    )

    st.markdown(
        "**1 hour ahead**"
    )


    st.caption(
        "Best research model"
    )

    st.markdown(
        "**Voting Regressor**"
    )


# ============================================================
# Load shared project data
# ============================================================

try:

    raw_df = get_raw_data()

    analysis_df = get_analysis_data()

except Exception as error:

    st.error(
        "The project dataset could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# PAGE 1 — Overview
# ============================================================

if selected_page == "Overview":

    best_summary = (
        load_result_csv(
            "best_model_summary.csv"
        )
        .iloc[0]
    )


    metrics_df = load_result_csv(
        "model_metrics.csv"
    )


    baseline_row = (
        metrics_df.loc[
            metrics_df[
                "Model"
            ]
            == "Naive Baseline"
        ]
        .iloc[0]
    )


    rmse_improvement = (
        (
            baseline_row[
                "RMSE"
            ]
            - best_summary[
                "RMSE"
            ]
        )
        / baseline_row[
            "RMSE"
        ]
        * 100
    )


    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    st.markdown(
        '<div class="red-line"></div>',
        unsafe_allow_html=True,
    )


    st.title(
        "Urban Electricity Consumption Forecasting"
    )


    st.write(
        """
Forecasting total city electricity demand one hour ahead using
weather conditions, time patterns, and recent consumption data.
        """
    )


    st.write("")


    # --------------------------------------------------------
    # Project summary
    # --------------------------------------------------------

    create_metric_row(
        [
            (
                "Dataset Records",
                f"{len(raw_df):,}",
            ),
            (
                "Electricity Zones",
                "3",
            ),
            (
                "Forecast Horizon",
                "1 Hour Ahead",
            ),
            (
                "Best Model",
                str(
                    best_summary[
                        "Best_Model"
                    ]
                ),
            ),
        ]
    )


    st.write("")
    st.write("")


    # --------------------------------------------------------
    # Problem and project approach
    # --------------------------------------------------------

    left_column, right_column = (
        st.columns(2)
    )


    with left_column:

        st.subheader(
            "Problem"
        )

        st.write(
            """
Electricity consumption changes throughout the day. Weather,
time patterns, and recent demand all affect how much electricity
a city requires.

Accurate short-term forecasts can support supply planning and
preparation for periods of high demand.
            """
        )


    with right_column:

        st.subheader(
            "Project Approach"
        )

        st.write(
            """
Historical electricity consumption, weather variables, time-based
features, and recent demand history were used to predict total
electricity consumption one hour ahead.

Several machine learning approaches were evaluated, including a
Voting Regressor ensemble.
            """
        )


    st.divider()


    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    st.subheader(
        "Best Model Performance"
    )

    st.caption(
        "Results from the unseen chronological test set."
    )


    create_metric_row(
        [
            (
                "MAE",
                f"{best_summary['MAE']:,.2f}",
            ),
            (
                "RMSE",
                f"{best_summary['RMSE']:,.2f}",
            ),
            (
                "R²",
                f"{best_summary['R2']:.4f}",
            ),
            (
                "MAPE",
                f"{best_summary['MAPE_Percent']:.2f}%",
            ),
        ]
    )


    st.write("")


    # --------------------------------------------------------
    # Final outcome
    # --------------------------------------------------------

    st.subheader(
        "Final Outcome"
    )


    outcome_left, outcome_right = (
        st.columns(2)
    )


    with outcome_left:

        st.metric(
            "Improvement Over Naive Baseline",
            f"{rmse_improvement:.1f}% lower RMSE",
        )

        st.write(
            """
The Voting Regressor achieved the strongest overall forecasting
performance among the evaluated approaches.
            """
        )


    with outcome_right:

        st.metric(
            "Unseen Test Records",
            "10,481",
        )

        st.write(
            """
The final evaluation was performed on later observations that
were not used during model training.
            """
        )


    st.divider()


    # --------------------------------------------------------
    # Project workflow
    # --------------------------------------------------------

    st.subheader(
        "Project Workflow"
    )


    workflow_columns = st.columns(
        5
    )


    workflow_items = [
        (
            "1",
            "Prepare Data",
            "Validate the secondary dataset.",
        ),
        (
            "2",
            "Create Features",
            "Build time and demand-history variables.",
        ),
        (
            "3",
            "Train Models",
            "Compare individual ML approaches.",
        ),
        (
            "4",
            "Build Ensemble",
            "Combine the strongest models.",
        ),
        (
            "5",
            "Evaluate",
            "Compare predictions with actual test values.",
        ),
    ]


    for column, (
        number,
        title,
        description,
    ) in zip(
        workflow_columns,
        workflow_items,
    ):

        with column:

            st.markdown(
                f"""
**{number}. {title}**

{description}
                """
            )


# ============================================================
# PAGE 2 — Dataset Explorer
# ============================================================

elif selected_page == "Dataset Explorer":

    page_header(
        "Dataset Explorer",
        (
            "Explore the secondary Tetouan City electricity "
            "consumption dataset used in this project."
        ),
    )


    create_metric_row(
        [
            (
                "Rows",
                f"{len(raw_df):,}",
            ),
            (
                "Columns",
                str(
                    raw_df.shape[1]
                ),
            ),
            (
                "Missing Values",
                str(
                    int(
                        raw_df
                        .isna()
                        .sum()
                        .sum()
                    )
                ),
            ),
            (
                "Duplicate Rows",
                str(
                    int(
                        raw_df
                        .duplicated()
                        .sum()
                    )
                ),
            ),
        ]
    )


    st.write("")


    start_date = (
        raw_df[
            "DateTime"
        ]
        .min()
    )


    end_date = (
        raw_df[
            "DateTime"
        ]
        .max()
    )


    st.markdown(
        f"""
**Dataset:** Tetouan City Power Consumption  
**Source type:** Secondary dataset  
**Recording interval:** 10 minutes  
**Dataset period:** {start_date.date()} → {end_date.date()}
        """
    )


    tab_preview, tab_summary, tab_columns = (
        st.tabs(
            [
                "Data Preview",
                "Summary Statistics",
                "Column Information",
            ]
        )
    )


    with tab_preview:

        st.dataframe(
            raw_df.head(25),
            use_container_width=True,
            hide_index=True,
        )


    with tab_summary:

        st.dataframe(
            raw_df.describe().T,
            use_container_width=True,
        )


    with tab_columns:

        column_information = pd.DataFrame(
            {
                "Column": [
                    "DateTime",
                    "Temperature",
                    "Humidity",
                    "Wind Speed",
                    "general diffuse flows",
                    "diffuse flows",
                    "Zone 1 Power Consumption",
                    "Zone 2 Power Consumption",
                    "Zone 3 Power Consumption",
                ],

                "Description": [
                    "Timestamp recorded every 10 minutes",
                    "Ambient temperature",
                    "Relative humidity",
                    "Wind speed",
                    "General diffuse solar radiation",
                    "Diffuse solar radiation",
                    "Electricity consumption in Zone 1",
                    "Electricity consumption in Zone 2",
                    "Electricity consumption in Zone 3",
                ],
            }
        )


        st.dataframe(
            column_information,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 3 — Consumption Analysis
# ============================================================

elif selected_page == "Consumption Analysis":

    page_header(
        "Consumption Analysis",
        (
            "Visualize electricity demand patterns across "
            "time, weather conditions, and zones."
        ),
    )


    total_column = (
        TOTAL_CONSUMPTION_COLUMN
    )


    daily_average = (
        analysis_df
        .set_index(
            "DateTime"
        )[total_column]
        .resample(
            "D"
        )
        .mean()
        .reset_index()
    )


    hourly_average = (
        analysis_df
        .groupby(
            "hour",
            as_index=False,
        )[total_column]
        .mean()
    )


    day_average = (
        analysis_df
        .groupby(
            "day_name",
            as_index=False,
        )[total_column]
        .mean()
    )


    day_average[
        "day_name"
    ] = pd.Categorical(
        day_average[
            "day_name"
        ],
        categories=DAY_ORDER,
        ordered=True,
    )


    day_average = (
        day_average
        .sort_values(
            "day_name"
        )
    )


    # --------------------------------------------------------
    # Consumption over time
    # --------------------------------------------------------

    st.subheader(
        "Electricity Consumption Over Time"
    )


    figure = px.line(
        daily_average,
        x="DateTime",
        y=total_column,

        labels={
            "DateTime":
                "Date",

            total_column:
                "Average Total Consumption",
        },
    )


    figure.update_traces(
        line={
            "color": PRIMARY_RED,
            "width": 2.5,
        }
    )


    figure = style_figure(
        figure,
        height=440,
    )


    st.plotly_chart(
        figure,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Hour and day
    # --------------------------------------------------------

    chart_left, chart_right = (
        st.columns(2)
    )


    with chart_left:

        st.subheader(
            "Average Hourly Consumption"
        )


        figure = px.bar(
            hourly_average,
            x="hour",
            y=total_column,

            labels={
                "hour":
                    "Hour of Day",

                total_column:
                    "Average Total Consumption",
            },
        )


        figure.update_traces(
            marker_color=PRIMARY_RED,
            marker_line_width=0,
        )


        figure = style_figure(
            figure,
            height=390,
        )


        st.plotly_chart(
            figure,
            use_container_width=True,
        )


    with chart_right:

        st.subheader(
            "Average Consumption by Day"
        )


        figure = px.bar(
            day_average,
            x="day_name",
            y=total_column,

            labels={
                "day_name":
                    "Day",

                total_column:
                    "Average Total Consumption",
            },
        )


        figure.update_traces(
            marker_color=PRIMARY_RED,
            marker_line_width=0,
        )


        figure = style_figure(
            figure,
            height=390,
        )


        st.plotly_chart(
            figure,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Zone comparison
    # --------------------------------------------------------

    st.subheader(
        "Electricity Zone Comparison"
    )


    zone_means = pd.DataFrame(
        {
            "Zone": [
                "Zone 1",
                "Zone 2",
                "Zone 3",
            ],

            "Average Consumption": [
                analysis_df[
                    "Zone 1 Power Consumption"
                ].mean(),

                analysis_df[
                    "Zone 2  Power Consumption"
                ].mean(),

                analysis_df[
                    "Zone 3  Power Consumption"
                ].mean(),
            ],
        }
    )


    figure = px.bar(
        zone_means,
        x="Zone",
        y="Average Consumption",
        color="Zone",

        color_discrete_sequence=[
            PRIMARY_RED,
            LIGHT_RED,
            DARK_RED,
        ],
    )


    figure.update_layout(
        showlegend=False,
    )


    figure = style_figure(
        figure,
        height=410,
    )


    st.plotly_chart(
        figure,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Temperature and weekend comparison
    # --------------------------------------------------------

    chart_left, chart_right = (
        st.columns(2)
    )


    with chart_left:

        st.subheader(
            "Temperature vs Consumption"
        )


        scatter_sample = (
            analysis_df
            .sample(
                min(
                    2500,
                    len(
                        analysis_df
                    ),
                ),
                random_state=42,
            )
        )


        figure = px.scatter(
            scatter_sample,
            x="Temperature",
            y=total_column,
            opacity=0.45,

            labels={
                total_column:
                    "Total Consumption",
            },
        )


        figure.update_traces(
            marker={
                "color": PRIMARY_RED,
                "size": 6,
            }
        )


        figure = style_figure(
            figure,
            height=400,
        )


        st.plotly_chart(
            figure,
            use_container_width=True,
        )


    with chart_right:

        st.subheader(
            "Weekday vs Weekend"
        )


        weekday_weekend = pd.DataFrame(
            {
                "Period": [
                    "Weekday",
                    "Weekend",
                ],

                "Average Consumption": [
                    analysis_df.loc[
                        analysis_df[
                            "is_weekend"
                        ]
                        == 0,
                        total_column,
                    ]
                    .mean(),

                    analysis_df.loc[
                        analysis_df[
                            "is_weekend"
                        ]
                        == 1,
                        total_column,
                    ]
                    .mean(),
                ],
            }
        )


        figure = px.bar(
            weekday_weekend,
            x="Period",
            y="Average Consumption",
            color="Period",

            color_discrete_sequence=[
                PRIMARY_RED,
                LIGHT_RED,
            ],
        )


        figure.update_layout(
            showlegend=False,
        )


        figure = style_figure(
            figure,
            height=400,
        )


        st.plotly_chart(
            figure,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    st.subheader(
        "Correlation Heatmap"
    )


    correlation_columns = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "general diffuse flows",
        "diffuse flows",
        "Zone 1 Power Consumption",
        "Zone 2  Power Consumption",
        "Zone 3  Power Consumption",
        total_column,
    ]


    correlation_matrix = (
        analysis_df[
            correlation_columns
        ]
        .corr()
    )


    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",

        color_continuous_scale=[
            [
                0.0,
                "#17171b",
            ],
            [
                0.25,
                "#3b1c1c",
            ],
            [
                0.5,
                "#81302c",
            ],
            [
                0.75,
                "#c9271f",
            ],
            [
                1.0,
                "#f9423a",
            ],
        ],
    )


    figure.update_layout(
        coloraxis_colorbar={
            "title":
                "Correlation",
        }
    )


    figure = style_figure(
        figure,
        height=620,
    )


    st.plotly_chart(
        figure,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Findings
    # --------------------------------------------------------

    st.subheader(
        "Key Findings"
    )


    peak_hour = int(
        hourly_average.loc[
            hourly_average[
                total_column
            ]
            .idxmax(),
            "hour",
        ]
    )


    lowest_hour = int(
        hourly_average.loc[
            hourly_average[
                total_column
            ]
            .idxmin(),
            "hour",
        ]
    )


    temperature_correlation = (
        analysis_df[
            "Temperature"
        ]
        .corr(
            analysis_df[
                total_column
            ]
        )
    )


    weekday_average = (
        analysis_df.loc[
            analysis_df[
                "is_weekend"
            ]
            == 0,
            total_column,
        ]
        .mean()
    )


    weekend_average = (
        analysis_df.loc[
            analysis_df[
                "is_weekend"
            ]
            == 1,
            total_column,
        ]
        .mean()
    )


    st.markdown(
        f"""
- Average demand peaks around **{peak_hour}:00** and is lowest
  around **{lowest_hour}:00**.
- **Zone 1** has the highest average electricity consumption.
- Average weekday demand is approximately
  **{weekday_average:,.0f}**, compared with
  **{weekend_average:,.0f}** on weekends.
- Temperature has a correlation of approximately
  **{temperature_correlation:.2f}** with total electricity consumption.
- Electricity demand varies substantially throughout the year.
        """
    )


# ============================================================
# PAGE 4 — Model Results
# ============================================================

elif selected_page == "Model Results":

    page_header(
        "Model Results",
        (
            "Compare the forecasting approaches evaluated "
            "during model development."
        ),
    )


    metrics_df = (
        load_result_csv(
            "model_metrics.csv"
        )
        .sort_values(
            "RMSE"
        )
        .reset_index(
            drop=True
        )
    )


    best_summary = (
        load_result_csv(
            "best_model_summary.csv"
        )
        .iloc[0]
    )


    feature_importance = (
        load_result_csv(
            "feature_importance.csv"
        )
        .sort_values(
            "Importance",
            ascending=False,
        )
    )


    create_metric_row(
        [
            (
                "Best Model",
                str(
                    best_summary[
                        "Best_Model"
                    ]
                ),
            ),
            (
                "RMSE",
                f"{best_summary['RMSE']:,.2f}",
            ),
            (
                "R²",
                f"{best_summary['R2']:.4f}",
            ),
            (
                "MAPE",
                f"{best_summary['MAPE_Percent']:.2f}%",
            ),
        ]
    )


    st.write("")


    # --------------------------------------------------------
    # Approach comparison
    # --------------------------------------------------------

    st.subheader(
        "Approach Comparison"
    )


    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )


    figure = px.bar(
        metrics_df,
        x="Model",
        y="RMSE",
    )


    figure.update_traces(
        marker_color=PRIMARY_RED,
        marker_line_width=0,
    )


    figure = style_figure(
        figure,
        title="RMSE Comparison",
        height=440,
    )


    st.plotly_chart(
        figure,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Performance interpretation
    # --------------------------------------------------------

    st.subheader(
        "Model Comparison Summary"
    )


    best_model_name = (
        metrics_df
        .iloc[0][
            "Model"
        ]
    )


    worst_model_name = (
        metrics_df
        .iloc[-1][
            "Model"
        ]
    )


    st.markdown(
        f"""
- **{best_model_name}** achieved the lowest RMSE and the best
  overall forecasting performance.
- The ensemble slightly outperformed the strongest individual models.
- **{worst_model_name}** had the highest RMSE among the evaluated
  approaches.
- The machine learning approaches substantially outperformed the
  naive persistence baseline.
        """
    )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance"
    )


    figure = px.bar(
        feature_importance.head(14),
        x="Importance",
        y="Feature",
        orientation="h",
    )


    figure.update_traces(
        marker_color=PRIMARY_RED,
        marker_line_width=0,
    )


    figure.update_layout(
        yaxis={
            "categoryorder":
                "total ascending",
        }
    )


    figure = style_figure(
        figure,
        height=560,
    )


    st.plotly_chart(
        figure,
        use_container_width=True,
    )


    st.info(
        """
Current total power consumption is the strongest predictor.
Recent demand history, solar-radiation variables, and time of
day also contribute to the one-hour-ahead forecast.
        """
    )


# ============================================================
# PAGE 5 — Prediction Results
# ============================================================

elif selected_page == "Prediction Results":

    page_header(
        "Prediction Results",
        (
            "Evaluate the best ensemble model on the unseen "
            "chronological test set."
        ),
    )


    predictions = (
        load_result_csv(
            "test_predictions.csv"
        )
        .copy()
    )


    predictions[
        "DateTime"
    ] = pd.to_datetime(
        predictions[
            "DateTime"
        ]
    )


    best_summary = (
        load_result_csv(
            "best_model_summary.csv"
        )
        .iloc[0]
    )


    if (
        "Absolute_Error"
        not in predictions.columns
    ):

        predictions[
            "Absolute_Error"
        ] = (
            predictions[
                "Actual"
            ]
            - predictions[
                "Predicted"
            ]
        ).abs()


    if (
        "Error"
        not in predictions.columns
    ):

        predictions[
            "Error"
        ] = (
            predictions[
                "Predicted"
            ]
            - predictions[
                "Actual"
            ]
        )


    # --------------------------------------------------------
    # Test-set summary
    # --------------------------------------------------------

    create_metric_row(
        [
            (
                "Test Records",
                f"{len(predictions):,}",
            ),
            (
                "R²",
                f"{best_summary['R2']:.4f}",
            ),
            (
                "RMSE",
                f"{best_summary['RMSE']:,.2f}",
            ),
            (
                "MAPE",
                f"{best_summary['MAPE_Percent']:.2f}%",
            ),
        ]
    )


    st.write("")


    st.info(
        """
The model was trained on the earlier 80% of the time-ordered data
and evaluated on the later 20%. The observations shown on this page
were not used during model training.
        """
    )


    # --------------------------------------------------------
    # Actual vs predicted timeline
    # --------------------------------------------------------

    st.subheader(
        "Actual vs Predicted Consumption"
    )


    display_options = {
        "Full test period":
            None,

        "First 7 days":
            7,

        "First 14 days":
            14,

        "First 30 days":
            30,
    }


    selected_period = st.selectbox(
        "Viewing period",
        options=list(
            display_options.keys()
        ),
    )


    selected_days = (
        display_options[
            selected_period
        ]
    )


    if selected_days is None:

        timeline_data = (
            predictions
            .set_index(
                "DateTime"
            )[
                [
                    "Actual",
                    "Predicted",
                ]
            ]
            .resample(
                "D"
            )
            .mean()
            .reset_index()
        )

        chart_note = (
            "Daily averages are shown for the full test period "
            "to keep the chart readable."
        )

    else:

        start_timestamp = (
            predictions[
                "DateTime"
            ]
            .min()
        )

        end_timestamp = (
            start_timestamp
            + pd.Timedelta(
                days=selected_days
            )
        )


        timeline_data = (
            predictions.loc[
                predictions[
                    "DateTime"
                ]
                < end_timestamp
            ][
                [
                    "DateTime",
                    "Actual",
                    "Predicted",
                ]
            ]
            .copy()
        )


        chart_note = (
            "Individual 10-minute test observations are shown."
        )


    st.caption(
        chart_note
    )


    figure = go.Figure()


    figure.add_trace(
        go.Scatter(
            x=timeline_data[
                "DateTime"
            ],

            y=timeline_data[
                "Actual"
            ],

            mode="lines",

            name="Actual",

            line={
                "color":
                    COMPARISON_BLUE,

                "width":
                    2.2,
            },
        )
    )


    figure.add_trace(
        go.Scatter(
            x=timeline_data[
                "DateTime"
            ],

            y=timeline_data[
                "Predicted"
            ],

            mode="lines",

            name="Predicted",

            line={
                "color":
                    PRIMARY_RED,

                "width":
                    2.2,
            },
        )
    )


    figure = style_figure(
        figure,
        height=500,
    )


    st.plotly_chart(
        figure,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Scatter and error distribution
    # --------------------------------------------------------

    chart_left, chart_right = (
        st.columns(2)
    )


    with chart_left:

        st.subheader(
            "Actual vs Predicted Scatter"
        )


        scatter_sample = (
            predictions
            .sample(
                min(
                    3000,
                    len(
                        predictions
                    ),
                ),
                random_state=42,
            )
        )


        figure = px.scatter(
            scatter_sample,
            x="Actual",
            y="Predicted",
            opacity=0.45,
        )


        figure.update_traces(
            marker={
                "color":
                    PRIMARY_RED,

                "size":
                    6,
            }
        )


        minimum_value = min(
            scatter_sample[
                "Actual"
            ]
            .min(),

            scatter_sample[
                "Predicted"
            ]
            .min(),
        )


        maximum_value = max(
            scatter_sample[
                "Actual"
            ]
            .max(),

            scatter_sample[
                "Predicted"
            ]
            .max(),
        )


        figure.add_trace(
            go.Scatter(
                x=[
                    minimum_value,
                    maximum_value,
                ],

                y=[
                    minimum_value,
                    maximum_value,
                ],

                mode="lines",

                name="Perfect Prediction",

                line={
                    "color":
                        COMPARISON_BLUE,

                    "dash":
                        "dash",
                },
            )
        )


        figure = style_figure(
            figure,
            height=430,
        )


        st.plotly_chart(
            figure,
            use_container_width=True,
        )


    with chart_right:

        st.subheader(
            "Prediction Error Distribution"
        )


        figure = px.histogram(
            predictions,
            x="Error",
            nbins=45,
        )


        figure.update_traces(
            marker_color=PRIMARY_RED,
            marker_line_width=0,
        )


        figure.add_vline(
            x=0,

            line_color=COMPARISON_BLUE,

            line_dash="dash",

            annotation_text="Zero error",

            annotation_font_color=COMPARISON_BLUE,
        )


        figure = style_figure(
            figure,
            height=430,
        )


        st.plotly_chart(
            figure,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Sample predictions
    # --------------------------------------------------------

    st.subheader(
        "Sample Test Predictions"
    )


    sample_size = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )


    sample_predictions = (
        predictions[
            [
                "DateTime",
                "Actual",
                "Predicted",
                "Absolute_Error",
            ]
        ]
        .head(
            sample_size
        )
        .copy()
    )


    sample_predictions[
        "Actual"
    ] = (
        sample_predictions[
            "Actual"
        ]
        .round(2)
    )


    sample_predictions[
        "Predicted"
    ] = (
        sample_predictions[
            "Predicted"
        ]
        .round(2)
    )


    sample_predictions[
        "Absolute_Error"
    ] = (
        sample_predictions[
            "Absolute_Error"
        ]
        .round(2)
    )


    sample_predictions = (
        sample_predictions
        .rename(
            columns={
                "Absolute_Error":
                    "Absolute Error",
            }
        )
    )


    st.dataframe(
        sample_predictions,
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # Largest errors
    # --------------------------------------------------------

    with st.expander(
        "View the 10 largest prediction errors"
    ):

        largest_errors = (
            predictions
            .nlargest(
                10,
                "Absolute_Error",
            )[
                [
                    "DateTime",
                    "Actual",
                    "Predicted",
                    "Absolute_Error",
                ]
            ]
            .copy()
        )


        largest_errors = (
            largest_errors
            .rename(
                columns={
                    "Absolute_Error":
                        "Absolute Error",
                }
            )
        )


        st.dataframe(
            largest_errors,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader(
        "Result Interpretation"
    )


    st.markdown(
        f"""
- The best model explains approximately
  **{best_summary['R2'] * 100:.1f}%** of the variation in unseen
  electricity consumption.
- The average percentage error is approximately
  **{best_summary['MAPE_Percent']:.2f}%**.
- The actual and predicted lines follow similar demand patterns
  throughout the test period.
- Most scatter points lie close to the perfect-prediction diagonal.
- The test-set results show that the ensemble generalizes well to
  later, unseen observations.
        """
    )