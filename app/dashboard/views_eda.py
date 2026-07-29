"""
Streamlit Exploratory Data Analysis (EDA) View with Plotly Charts.
"""

import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

DATASET_API_URL = "http://localhost:8000/api/v1/dataset"


def render_eda_view():
    """Renders automated summary metrics, correlation heatmaps, missing value reports, and outlier charts."""
    st.title("📊 Exploratory Data Analysis")

    dataset_id = st.session_state.get("dataset_id")
    if not dataset_id:
        st.warning("No active dataset found. Please upload a dataset on the Dataset Management page.")
        return

    token = st.session_state.get("access_token")
    headers = {"Authorization": f"Bearer {token}"}

    with st.spinner("Generating EDA report from backend..."):
        try:
            res = requests.get(f"{DATASET_API_URL}/{dataset_id}/summary", headers=headers)
            if res.status_code != 200:
                st.error("Failed to load dataset EDA summary.")
                return
            eda_data = res.json()
        except Exception as e:
            st.error(f"Backend communication error: {str(e)}")
            return

    summary = eda_data["summary"]
    missing_report = eda_data["missing_report"]
    outlier_report = eda_data["outlier_report"]
    correlation = eda_data["correlation"]

    st.subheader("Dataset Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Rows", summary["total_rows"])
    m2.metric("Total Columns", summary["total_columns"])
    m3.metric("Memory Usage", f"{summary['memory_usage_bytes'] / (1024 * 1024):.2f} MB")

    # Missing Value Chart
    st.markdown("---")
    st.subheader("Missing Value Distribution")
    missing_df = pd.DataFrame(missing_report["missing_report_by_column"])
    if not missing_df.empty:
        fig_missing = px.bar(
            missing_df,
            x="column",
            y="missing_percentage",
            color="has_missing",
            title="Column Missing Percentage (%)",
            labels={"missing_percentage": "Missing %", "column": "Column Name"},
        )
        st.plotly_chart(fig_missing, use_container_width=True)

    # Correlation Heatmap
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    corr_cols = correlation.get("columns", [])
    corr_matrix = correlation.get("matrix", [])

    if corr_cols and corr_matrix:
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=corr_cols,
                y=corr_cols,
                colorscale="Viridis",
                zmin=-1,
                zmax=1,
            )
        )
        fig_corr.update_layout(title="Pearson Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Insufficient numeric features available for correlation computation.")

    # Outliers Summary
    st.markdown("---")
    st.subheader("Outlier Detection (IQR Rule)")
    outlier_df = pd.DataFrame(outlier_report["outlier_report_by_column"])
    if not outlier_df.empty:
        fig_outlier = px.bar(
            outlier_df,
            x="column",
            y="outlier_count",
            title="Detected Outliers Count per Column",
            labels={"outlier_count": "Outlier Count", "column": "Column Name"},
        )
        st.plotly_chart(fig_outlier, use_container_width=True)
