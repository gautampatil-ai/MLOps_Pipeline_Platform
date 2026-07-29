"""
Main Streamlit Dashboard Entrypoint Application with Sidebar Navigation.
"""

import streamlit as st

from app.dashboard.views_auth import render_auth_view
from app.dashboard.views_dataset import render_dataset_view
from app.dashboard.views_eda import render_eda_view
from app.dashboard.views_train import render_train_view
from app.dashboard.views_explain import render_explain_view
from app.dashboard.views_monitor import render_monitor_view

# Page Config
st.set_page_config(
    page_title="MLOps Pipeline Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Renders main Streamlit layout with sidebar navigation menu."""
    st.sidebar.title("🚀 MLOps Platform")
    st.sidebar.markdown("Production ML Pipeline Platform")

    # Session authentication status indicator
    if st.session_state.get("authenticated", False):
        st.sidebar.success(f"User: **{st.session_state.get('username')}**")
    else:
        st.sidebar.info("Status: Unauthenticated")

    st.sidebar.markdown("---")

    # Navigation choices
    pages = {
        "🔐 Authentication": render_auth_view,
        "📂 Dataset Management": render_dataset_view,
        "📊 Exploratory Data Analysis": render_eda_view,
        "🤖 Model Training": render_train_view,
        "💡 Explainable AI (SHAP)": render_explain_view,
        "📈 Performance & Drift": render_monitor_view,
    }

    selection = st.sidebar.radio("Navigation Menu", list(pages.keys()))

    # Render selected view page
    pages[selection]()


if __name__ == "__main__":
    main()
