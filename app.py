import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


DATA_FILE = "House Price Prediction Dataset.csv"
TARGET_COLUMN = "price"
IDENTIFIER_COLUMNS = {
    "id",
    "houseid",
    "homeid",
    "propertyid",
    "recordid",
    "serialno",
    "serialnumber",
    "index",
    "unnamed0",
}
YEAR_COLUMNS = {
    "year",
    "yearbuilt",
    "year_built",
    "builtyear",
    "built_year",
    "constructionyear",
    "construction_year",
}
PLOT_LAYOUT = {
    "template": "plotly_white",
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "font": {"color": "#111827", "size": 13},
    "title_font": {"color": "#111827", "size": 20},
    "xaxis": {"gridcolor": "#e5e7eb", "linecolor": "#9ca3af", "zerolinecolor": "#d1d5db"},
    "yaxis": {"gridcolor": "#e5e7eb", "linecolor": "#9ca3af", "zerolinecolor": "#d1d5db"},
}


st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            color-scheme: light !important;
        }
        :root {
            --app-bg: #f4f7fb;
            --panel-bg: #ffffff;
            --text-main: #111827;
            --text-muted: #4b5563;
            --border: #d7dee8;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
        }
        .stApp {
            background: var(--app-bg);
            color: var(--text-main);
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
            background: var(--app-bg);
        }
        [data-testid="stSidebar"] {
            background: var(--panel-bg);
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, label, span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text-main);
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetric"] * {
            color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }
        .main-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }
        .prediction-box {
            background: #12335f;
            color: #ffffff !important;
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 0.75rem;
        }
        .prediction-box,
        .prediction-box *,
        .prediction-box div,
        .prediction-box h3 {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }
        .prediction-box h3 {
            margin: 0;
            font-size: 1.65rem;
        }
        .dataset-banner {
            background: #e8f5ee;
            border: 1px solid #b8e2c8;
            color: #14532d;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 1rem 0;
            font-weight: 600;
        }
        .small-muted {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        .stDataFrame {
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] div,
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploader"] div,
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploader"] *,
        div[data-testid="stFileUploaderDropzone"] * {
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploaderDropzone"] button {
            background: #eff6ff !important;
            border: 1px solid #bfdbfe !important;
            color: #1d4ed8 !important;
            -webkit-text-fill-color: #1d4ed8 !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg {
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            fill: var(--text-main) !important;
        }
        div[data-baseweb="popover"],
        div[role="listbox"],
        ul[role="listbox"],
        li[role="option"] {
            background: #ffffff !important;
            color: var(--text-main) !important;
        }
        li[role="option"] *,
        div[role="listbox"] * {
            color: var(--text-main) !important;
        }
        li[role="option"]:hover {
            background: var(--accent-soft) !important;
        }
        div[data-testid="stNumberInput"] button {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] div,
        div[data-testid="stNumberInput"] span {
            background-color: #ffffff !important;
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            opacity: 1 !important;
        }
        button[data-baseweb="tab"] p {
            color: var(--text-muted) !important;
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--accent) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom-color: var(--accent) !important;
        }
        div[data-testid="stSegmentedControl"] {
            background: transparent !important;
        }
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stSegmentedControl"] label {
            background: #ffffff !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stSegmentedControl"] button *,
        div[data-testid="stSegmentedControl"] label * {
            color: var(--text-main) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
        div[data-testid="stSegmentedControl"] label[data-checked="true"] {
            background: var(--accent-soft) !important;
            border-color: var(--accent) !important;
        }
        .stSlider [data-baseweb="slider"] div {
            color: var(--text-main);
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .results-table th {
            background: #f1f5f9;
            color: #111827;
            text-align: left;
            padding: 0.8rem;
            border-bottom: 1px solid var(--border);
            font-weight: 700;
        }
        .results-table td {
            color: #111827;
            padding: 0.8rem;
            border-bottom: 1px solid #eef2f7;
        }
        .results-table td:not(:first-child) {
            text-align: right;
        }
        .table-wrap {
            max-height: 460px;
            overflow: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .light-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }
        .light-table th {
            position: sticky;
            top: 0;
            background: #f1f5f9;
            color: #111827;
            text-align: left;
            padding: 0.7rem;
            border-bottom: 1px solid var(--border);
            font-weight: 700;
            z-index: 1;
        }
        .light-table td {
            color: #111827;
            padding: 0.65rem 0.7rem;
            border-bottom: 1px solid #eef2f7;
            background: #ffffff;
        }
        .light-table tr:nth-child(even) td {
            background: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_column_name(column_name: str) -> str:
    cleaned = column_name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    return cleaned


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def discover_local_csv_files() -> list[Path]:
    csv_files = sorted(Path.cwd().glob("*.csv"))
    if not csv_files:
        return []

    preferred = [path for path in csv_files if path.name.lower() == DATA_FILE.lower()]
    remaining = [path for path in csv_files if path.name.lower() != DATA_FILE.lower()]
    return preferred + remaining


def find_identifier_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col in IDENTIFIER_COLUMNS]


def is_year_column(column_name: str) -> bool:
    return column_name in YEAR_COLUMNS or (
        "year" in column_name and ("built" in column_name or "construction" in column_name)
    )


def clean_year_values(series: pd.Series) -> pd.Series:
    numeric_years = pd.to_numeric(series, errors="coerce")
    numeric_years = numeric_years.where(
        (numeric_years >= 1000) & (numeric_years <= 9999)
    )
    median_year = numeric_years.median()
    if pd.isna(median_year):
        median_year = 2000
    return np.floor(numeric_years.fillna(median_year)).astype(int)


def prepare_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    df = raw_df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    df = df.drop_duplicates()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset must contain a '{TARGET_COLUMN}' column after cleaning column names."
        )

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
            if is_year_column(col):
                df[col] = clean_year_values(df[col])
        elif is_year_column(col):
            df[col] = clean_year_values(df[col])
        else:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
            df[col] = df[col].fillna("Unknown")

    ignored_columns = find_identifier_columns(df)
    model_df = df.drop(columns=ignored_columns)

    encoded_df = pd.get_dummies(model_df, drop_first=True, dtype=int)
    X = encoded_df.drop(TARGET_COLUMN, axis=1)
    y = encoded_df[TARGET_COLUMN]
    return df, model_df, X, y, ignored_columns


@st.cache_resource(show_spinner=False)
def train_models(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=150, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    results = {
        "Linear Regression": {
            "model": lr_model,
            "predictions": lr_predictions,
            "rmse": mean_squared_error(y_test, lr_predictions) ** 0.5,
            "mae": mean_absolute_error(y_test, lr_predictions),
            "r2": r2_score(y_test, lr_predictions),
        },
        "Random Forest": {
            "model": rf_model,
            "predictions": rf_predictions,
            "rmse": mean_squared_error(y_test, rf_predictions) ** 0.5,
            "mae": mean_absolute_error(y_test, rf_predictions),
            "r2": r2_score(y_test, rf_predictions),
        },
    }

    cv_score = np.nan
    if len(X) >= 5:
        cv_score = cross_val_score(rf_model, X, y, cv=5, scoring="r2").mean()

    return X_test, y_test, results, cv_score


def money(value: float) -> str:
    return f"${value:,.2f}"


def render_light_table(df: pd.DataFrame, max_rows: int | None = None) -> None:
    table_df = df.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)

    html = table_df.to_html(
        index=False,
        classes="light-table",
        border=0,
        escape=True,
    )
    st.markdown(f'<div class="table-wrap">{html}</div>', unsafe_allow_html=True)


def build_prediction_input(df: pd.DataFrame) -> pd.DataFrame:
    input_values = {}
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    for col in feature_columns:
        readable = col.replace("_", " ").title()
        if pd.api.types.is_numeric_dtype(df[col]):
            if is_year_column(col):
                min_year = int(max(1000, np.floor(df[col].min())))
                max_year = int(min(9999, np.floor(df[col].max())))
                median_year = int(np.floor(df[col].median()))

                if min_year >= max_year:
                    max_year = min_year + 1

                input_values[col] = st.number_input(
                    readable,
                    min_value=min_year,
                    max_value=max_year,
                    value=min(max(median_year, min_year), max_year),
                    step=1,
                    format="%d",
                )
            else:
                min_value = float(df[col].min())
                max_value = float(df[col].max())
                median_value = float(df[col].median())

                if np.isclose(min_value, max_value):
                    max_value = min_value + 1.0

                input_values[col] = st.number_input(
                    readable,
                    min_value=min_value,
                    max_value=max_value,
                    value=median_value,
                    step=max((max_value - min_value) / 100, 1.0),
                )
        else:
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            input_values[col] = st.selectbox(readable, options=options)

    return pd.DataFrame([input_values])


def align_input(input_df: pd.DataFrame, model_columns: pd.Index) -> pd.DataFrame:
    encoded_input = pd.get_dummies(input_df, dtype=int)
    return encoded_input.reindex(columns=model_columns, fill_value=0)


def actual_vs_predicted_chart(y_test: pd.Series, predictions: np.ndarray, model_name: str):
    limit = min(100, len(y_test))
    chart_df = pd.DataFrame(
        {
            "House Index": np.arange(1, limit + 1),
            "Actual Price": y_test.values[:limit],
            "Predicted Price": predictions[:limit],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["House Index"],
            y=chart_df["Actual Price"],
            mode="markers",
            name="Actual Price",
            marker=dict(color="#2563eb", size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["House Index"],
            y=chart_df["Predicted Price"],
            mode="markers",
            name="Predicted Price",
            marker=dict(color="#16a34a" if model_name == "Random Forest" else "#f97316", size=8),
        )
    )
    fig.update_layout(
        title=f"{model_name}: Actual vs Predicted Prices",
        xaxis_title="House Index",
        yaxis_title="Price",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOT_LAYOUT,
    )
    return fig


def metric_row(results: dict, cv_score: float):
    lr = results["Linear Regression"]
    rf = results["Random Forest"]
    cols = st.columns(4)
    cols[0].metric("Linear RMSE", money(lr["rmse"]))
    cols[1].metric("Random Forest RMSE", money(rf["rmse"]))
    cols[2].metric("Best R2 Score", f"{max(lr['r2'], rf['r2']):.3f}")
    cols[3].metric("RF Cross-Val R2", "N/A" if np.isnan(cv_score) else f"{cv_score:.3f}")


st.markdown('<div class="main-title">House Price Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Train Linear Regression and Random Forest models, compare performance, explore graphs, and predict a custom house price.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset")
    default_data_path = Path(DATA_FILE)

    if default_data_path.exists():
        selected_local_file = DATA_FILE
        st.caption(f"Default dataset loaded from `{DATA_FILE}`.")
    else:
        selected_local_file = None
        st.caption(f"Place `{DATA_FILE}` in the same folder as this app.")

    uploaded_file = st.file_uploader("Upload another CSV", type=["csv"])

    st.header("Training")
    test_size = st.slider("Test data size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)


raw_df = None
dataset_source = None

if uploaded_file is not None:
    raw_df = load_csv_from_bytes(uploaded_file.getvalue())
    dataset_source = uploaded_file.name
elif selected_local_file:
    local_data_path = Path(selected_local_file)
    raw_df = load_csv_from_path(str(local_data_path))
    dataset_source = selected_local_file

if raw_df is None:
    st.info(
        f"Add your dataset CSV in this folder as `{DATA_FILE}`, or upload it from the sidebar."
    )
    st.stop()

st.markdown(
    f'<div class="dataset-banner">Dataset loaded: {dataset_source}</div>',
    unsafe_allow_html=True,
)

try:
    df, model_df, X, y, ignored_columns = prepare_data(raw_df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if len(df) < 10:
    st.error("Dataset needs at least 10 valid rows for train-test split and useful charts.")
    st.stop()

if X.shape[1] == 0:
    st.error("Dataset needs at least one feature column besides price.")
    st.stop()

with st.spinner("Training models and preparing dashboard..."):
    X_test, y_test, results, cv_score = train_models(X, y, test_size, int(random_state))

metric_row(results, cv_score)

tabs = st.tabs(["Predict", "Model Results", "Graphs", "Dataset"])

with tabs[0]:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Enter House Details")
        user_input_df = build_prediction_input(model_df)

    with right:
        st.subheader("Estimated Price")
        aligned_input = align_input(user_input_df, X.columns)

        lr_price = results["Linear Regression"]["model"].predict(aligned_input)[0]
        rf_price = results["Random Forest"]["model"].predict(aligned_input)[0]
        final_price = (lr_price + rf_price) / 2

        st.markdown(
            f"""
            <div class="prediction-box">
                <div>Estimated Market Value</div>
                <h3>{money(final_price)}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        col_a, col_b = st.columns(2)
        col_a.metric("Linear Regression", money(lr_price))
        col_b.metric("Random Forest", money(rf_price))

        st.caption(
            "Final estimate is the average of Linear Regression and Random Forest predictions."
        )

with tabs[1]:
    st.subheader("Model Performance")
    results_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "RMSE": model_result["rmse"],
                "MAE": model_result["mae"],
                "R2 Score": model_result["r2"],
            }
            for model_name, model_result in results.items()
        ]
    )
    table_rows = "\n".join(
        f"""
        <tr>
            <td>{row['Model']}</td>
            <td>{money(row['RMSE'])}</td>
            <td>{money(row['MAE'])}</td>
            <td>{row['R2 Score']:.3f}</td>
        </tr>
        """
        for _, row in results_df.iterrows()
    )
    st.markdown(
        f"""
        <table class="results-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R2 Score</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    fig = px.bar(
        results_df,
        x="Model",
        y="RMSE",
        color="Model",
        text=results_df["RMSE"].map(money),
        color_discrete_map={
            "Linear Regression": "#2563eb",
            "Random Forest": "#16a34a",
        },
        title="Model Performance Comparison by RMSE",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=430,
        yaxis_title="RMSE",
        margin=dict(l=20, r=20, t=60, b=20),
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    chart_choice = st.selectbox(
        "Select Graph",
        ["Linear Regression", "Random Forest", "Correlation Heatmap"],
        index=1,
    )

    if chart_choice in ["Linear Regression", "Random Forest"]:
        st.plotly_chart(
            actual_vs_predicted_chart(
                y_test,
                results[chart_choice]["predictions"],
                chart_choice,
            ),
            use_container_width=True,
        )
    else:
        corr_df = pd.get_dummies(df, drop_first=True, dtype=int).corr(numeric_only=True)
        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Heatmap",
            aspect="auto",
        )
        fig.update_layout(height=620, margin=dict(l=20, r=20, t=60, b=20))
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Cleaned Dataset Preview")
    stat_cols = st.columns(4)
    stat_cols[0].metric("Rows", f"{len(df):,}")
    stat_cols[1].metric("Columns", f"{len(df.columns):,}")
    stat_cols[2].metric("Features Used", f"{len(X.columns):,}")
    stat_cols[3].metric("Ignored ID Columns", f"{len(ignored_columns):,}")

    if ignored_columns:
        st.caption(
            "Ignored for prediction: "
            + ", ".join(col.replace("_", " ").title() for col in ignored_columns)
        )

    render_light_table(df, max_rows=100)

    with st.expander("Summary statistics"):
        summary_df = df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
        render_light_table(summary_df)
