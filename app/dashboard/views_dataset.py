"""
Streamlit Dataset Upload & Profiling View.
"""

import requests
import pandas as pd
import streamlit as st

DATASET_API_URL = "http://localhost:8000/api/v1/dataset"


def render_dataset_view():
    """Renders dataset file upload and automated validation view."""
    st.title("📂 Dataset Management & Ingestion")

    if not st.session_state.get("authenticated", False):
        st.warning("Please log in via the Authentication page to upload datasets.")
        return

    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if st.button("Ingest Dataset"):
            token = st.session_state.get("access_token")
            headers = {"Authorization": f"Bearer {token}"}
            
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            with st.spinner("Processing and parsing dataset..."):
                try:
                    res = requests.post(f"{DATASET_API_URL}/upload", files=files, headers=headers)
                    if res.status_code == 201:
                        ds_data = res.json()
                        st.session_state["dataset_id"] = ds_data["id"]
                        st.session_state["dataset_filename"] = ds_data["filename"]
                        st.success(f"Dataset successfully ingested! ID: {ds_data['id']}")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Rows", ds_data["row_count"])
                        col2.metric("Columns", ds_data["column_count"])
                        col3.metric("Size (Bytes)", ds_data["file_size_bytes"])
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    st.markdown("---")
    st.subheader("Active Session Dataset Preview")

    if "dataset_id" in st.session_state:
        st.info(f"Currently active Dataset ID: **{st.session_state['dataset_id']}** ({st.session_state.get('dataset_filename', '')})")
    else:
        st.warning("No dataset uploaded in this session yet.")
