"""
Streamlit Authentication & User Session Management View.
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1/auth"


def render_auth_view():
    """Renders Login and Registration tabs for user authentication."""
    st.title("🔐 Authentication & User Portal")

    if st.session_state.get("authenticated", False):
        st.success(f"Logged in as **{st.session_state.get('username', 'User')}**")
        if st.button("Logout", type="primary"):
            st.session_state["authenticated"] = False
            st.session_state["access_token"] = None
            st.session_state["username"] = None
            st.rerun()
        return

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to MLOps Platform")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login", key="btn_login"):
            if not username or not password:
                st.warning("Please provide both username and password.")
                return

            try:
                res = requests.post(
                    f"{API_URL}/login",
                    data={"username": username, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["authenticated"] = True
                    st.session_state["access_token"] = data["access_token"]
                    st.session_state["username"] = username
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
            except Exception as e:
                st.error(f"Failed to connect to backend server: {str(e)}")

    with tab2:
        st.subheader("Create New Account")
        reg_username = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pwd")

        if st.button("Register", key="btn_reg"):
            if not reg_username or not reg_email or not reg_password:
                st.warning("Please fill in all registration fields.")
                return

            try:
                res = requests.post(
                    f"{API_URL}/register",
                    json={"username": reg_username, "email": reg_email, "password": reg_password},
                )
                if res.status_code == 201:
                    st.success("Account created successfully! You can now log in.")
                else:
                    err_msg = res.json().get("detail", "Registration failed.")
                    st.error(f"Registration Error: {err_msg}")
            except Exception as e:
                st.error(f"Backend connection error: {str(e)}")
