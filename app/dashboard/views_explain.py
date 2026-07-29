"""
Streamlit Model Explainability (SHAP) View.
"""

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app.core.config import settings
from app.explainability.shap_engine import SHAPExplainer


def render_explain_view():
    """Renders SHAP global feature importances and local single-prediction waterfall plots."""
    st.title("💡 Explainable AI & Model Interpretability")

    model_path = settings.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        st.warning("No production model found. Train a model first before requesting explanations.")
        return

    st.info("Loaded active Production Model artifact (`best_model.joblib`).")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load production model artifact: {str(e)}")
        return

    st.subheader("Upload Sample Data for Explanation")
    uploaded_file = st.file_uploader("Upload CSV sample data to calculate SHAP values", type=["csv"], key="shap_upload")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        if st.button("Generate SHAP Global Feature Importance"):
            with st.spinner("Computing SHAP values..."):
                try:
                    explainer = SHAPExplainer(model=model, background_data=df.head(100))
                    summary = explainer.get_global_feature_importance(df)

                    st.markdown("---")
                    st.subheader("Global Feature Importance Summary")
                    st.json(summary)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    if explainer.shap_values is not None:
                        import shap
                        shap.summary_plot(explainer.shap_values, df, show=False)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"SHAP calculation error: {str(e)}")
