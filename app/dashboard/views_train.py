"""
Streamlit Automated Model Training & Comparison Leaderboard View.
"""

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

TRAIN_API_URL = "http://localhost:8000/api/v1/train"


def render_train_view():
    """Renders automated training trigger configuration and model leaderboard."""
    st.title("🤖 Automated Machine Learning Pipeline")

    dataset_id = st.session_state.get("dataset_id")
    if not dataset_id:
        st.warning("No dataset selected. Please upload a dataset first.")
        return

    token = st.session_state.get("access_token")
    headers = {"Authorization": f"Bearer {token}"}

    st.subheader("Pipeline Settings")
    target_column = st.text_input("Target Column Name", help="Name of target label column in dataset")
    problem_type = st.selectbox("Problem Type Override", options=["auto", "classification", "regression"])
    perform_tuning = st.checkbox("Enable Hyperparameter Optimization (Grid / Random Search)", value=False)

    if st.button("Launch Training Pipeline", type="primary"):
        if not target_column:
            st.warning("Please enter the target column name.")
            return

        payload = {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "problem_type": None if problem_type == "auto" else problem_type,
            "perform_tuning": perform_tuning,
        }

        with st.spinner("Training candidate models across registry algorithms..."):
            try:
                res = requests.post(f"{TRAIN_API_URL}/execute", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Pipeline Execution Complete! Best Model: **{data['best_model_name']}**")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Task Type", data["problem_type"].upper())
                    c2.metric("Best Model", data["best_model_name"])
                    c3.metric("Top Score", f"{data['best_score']:.4f}")

                    st.markdown("---")
                    st.subheader("Model Comparison Leaderboard")
                    leaderboard_df = pd.DataFrame(data["leaderboard"])
                    st.dataframe(leaderboard_df, use_container_width=True)

                    primary_metric = "f1_score" if data["problem_type"] == "classification" else "r2_score"
                    if primary_metric in leaderboard_df.columns:
                        fig = px.bar(
                            leaderboard_df,
                            x="Model Name",
                            y=primary_metric,
                            title=f"Algorithm Metric Comparison ({primary_metric})",
                            color=primary_metric,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.session_state["mlflow_run_id"] = data["mlflow_run_id"]
                else:
                    st.error(f"Training pipeline error: {res.text}")
            except Exception as e:
                st.error(f"Failed to communicate with training API: {str(e)}")
