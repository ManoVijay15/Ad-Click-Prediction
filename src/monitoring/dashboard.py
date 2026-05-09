"""Streamlit dashboard for ad-click prediction monitoring.

Reads from the predictions table populated by src/monitoring/logger.py.
Run:
    streamlit run src/monitoring/dashboard.py
"""

import json
import os

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st

PG_DSN = os.getenv(
    "PREDICTION_LOG_DSN",
    "postgresql://adclick:adclick@localhost:5432/adclick_mlflow",
)

st.set_page_config(page_title="Ad-Click Prediction Monitor", layout="wide")
st.title("Ad-Click Prediction — Live Monitor")


@st.cache_data(ttl=15)
def load_predictions(limit: int = 50_000) -> pd.DataFrame:
    try:
        with psycopg2.connect(PG_DSN) as conn:
            df = pd.read_sql(
                "SELECT id, ts, click_probability, will_click, model_version, "
                "request_payload, cached_features, actual_click "
                f"FROM predictions ORDER BY ts DESC LIMIT {limit}",
                conn,
            )
    except psycopg2.Error as e:
        st.error(f"Cannot connect to PostgreSQL: {e}")
        return pd.DataFrame()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


df = load_predictions()

if df.empty:
    st.warning("No predictions logged yet. Send a few /predict requests and refresh.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", f"{len(df):,}")
col2.metric("Predicted CTR", f"{df['click_probability'].mean():.2%}")
labeled = df[df["actual_click"].notna()]
col3.metric(
    "Actual CTR (with feedback)",
    f"{labeled['actual_click'].mean():.2%}" if len(labeled) else "no feedback yet",
)
col4.metric("Active model version", df["model_version"].iloc[0] or "—")

st.divider()

# Volume + CTR over time
left, right = st.columns(2)

with left:
    st.subheader("Prediction Volume Over Time")
    by_min = df.set_index("ts").resample("1min").size().rename("count").reset_index()
    fig = px.area(by_min, x="ts", y="count", labels={"ts": "Time", "count": "Predictions"})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Predicted CTR Over Time")
    by_min = (
        df.set_index("ts")
        .resample("1min")["click_probability"]
        .mean()
        .reset_index()
    )
    fig = px.line(by_min, x="ts", y="click_probability",
                  labels={"ts": "Time", "click_probability": "Mean predicted CTR"})
    fig.add_hline(y=0.17, line_dash="dash", line_color="gray",
                  annotation_text="Training base rate (17%)")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Distribution + version split
left, right = st.columns(2)

with left:
    st.subheader("Click Probability Distribution")
    fig = px.histogram(df, x="click_probability", nbins=40,
                       labels={"click_probability": "Predicted P(click)"})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Predictions by Model Version")
    by_ver = df.groupby("model_version").size().reset_index(name="count")
    fig = px.bar(by_ver, x="model_version", y="count",
                 labels={"model_version": "Version", "count": "Predictions"})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Cached feature signal — site CTR distribution
st.subheader("Cached Site CTR (from Redis feature store)")
site_ctrs = []
for cf in df["cached_features"].dropna():
    if isinstance(cf, str):
        cf = json.loads(cf)
    if cf and "site_ctr_7d" in cf:
        site_ctrs.append(cf["site_ctr_7d"])

if site_ctrs:
    fig = px.histogram(x=site_ctrs, nbins=30,
                       labels={"x": "Site CTR (7d)", "y": "Count"})
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No cached features yet — populate Redis with `make populate-store`.")

# Recent predictions table
st.subheader("Recent Predictions")
display_cols = ["id", "ts", "click_probability", "will_click", "model_version", "actual_click"]
st.dataframe(df[display_cols].head(50), use_container_width=True)

st.caption(f"Auto-refreshes every 15s · Source: {PG_DSN.split('@')[-1]}")
