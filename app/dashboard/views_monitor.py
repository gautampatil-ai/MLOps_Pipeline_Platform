"""
Streamlit Performance & Data Drift Monitoring View.
"""

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

MONITOR_API_URL = "http://localhost:8000/api/v1/monitor"


def render_monitor_view():
    """Renders data drift detector interface comparing baseline dataset against runtime inputs."""
    st.title("📈 Model Performance & Data Drift Monitoring")

    dataset_id = st.session_state.get("dataset_id")
    if not dataset_id:
        st.warning("No reference dataset found in session. Please upload a baseline dataset first.")
        return

    st.info(f"Baseline Reference Dataset ID: **{dataset_id}**")

    st.subheader("Upload Production Runtime Batch for Drift Analysis")
    uploaded_file = st.file_uploader("Upload incoming production batch CSV", type=["csv"], key="drift_upload")

    if uploaded_file is not None:
        curr_df = pd.read_csv(uploaded_file)
        st.write("Production Data Sample:", curr_df.head())

        if st.button("Analyze Data Drift"):
            payload = {
                "reference_dataset_id": dataset_id,
                "current_features": curr_df.to_dict(orient="records"),
            }

            with st.spinner("Running Kolmogorov-Smirnov statistical significance tests..."):
                try:
                    res = requests.post(f"{MONITOR_API_URL}/drift", json=payload)
                    if res.status_code == 200:
                        drift_res = res.json()
                        st.markdown("---")
                        st.subheader("Drift Detection Results")

                        if drift_res["has_drift"]:
                            st.error(f"⚠️ DATA DRIFT DETECTED in {len(drift_res['drifted_features'])} features!")
                            st.write("Drifted Features:", drift_res["drifted_features"])
                        else:
                            st.success("✅ No statistically significant data drift detected.")

                        # Plot P-Values
                        p_vals = drift_res["p_values"]
                        p_df = pd.DataFrame(list(p_vals.items()), columns=["Feature", "P-Value"])
                        
                        fig = px.bar(
                            p_df,
                            x="Feature",
                            y="P-Value",
                            title="Feature Drift Kolmogorov-Smirnov P-Values (Threshold = 0.05)",
                        )
                        fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Drift Threshold (0.05)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Drift monitoring request failed: {res.text}")
                except Exception as e:
                    st.error(f"Error communicating with monitoring API: {str(e)}")
