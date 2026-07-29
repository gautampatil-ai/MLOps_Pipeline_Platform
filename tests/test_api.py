"""
Integration Tests for FastAPI Endpoints.
"""

import pytest


def test_health_check_endpoint(client):
    """Tests platform health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_user_registration_and_login(client):
    """Tests register and login endpoints."""
    # 1. Register new user
    reg_payload = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "securepassword123",
    }
    reg_response = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_response.status_code == 201
    assert reg_response.json()["username"] == "newuser"

    # 2. Login with registered user
    login_payload = {
        "username": "newuser",
        "password": "securepassword123",
    }
    login_response = client.post(
        "/api/v1/auth/login",
        data=login_payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()


def test_dataset_upload_unauthorized(client, sample_csv_file):
    """Tests uploading a dataset without JWT token yields HTTP 401."""
    with open(sample_csv_file, "rb") as f:
        response = client.post(
            "/api/v1/dataset/upload",
            files={"file": ("sample.csv", f, "text/csv")},
        )
    assert response.status_code == 401


def test_dataset_upload_and_summary(client, auth_headers, sample_csv_file):
    """Tests uploading dataset and fetching its EDA summary."""
    # Upload
    with open(sample_csv_file, "rb") as f:
        upload_res = client.post(
            "/api/v1/dataset/upload",
            files={"file": ("sample.csv", f, "text/csv")},
            headers=auth_headers,
        )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["id"]

    # Retrieve EDA Summary
    summary_res = client.get(
        f"/api/v1/dataset/{dataset_id}/summary",
        headers=auth_headers,
    )
    assert summary_res.status_code == 200
    assert "summary" in summary_res.json()
    assert summary_res.json()["summary"]["total_rows"] == 100


def test_training_pipeline_execution(client, auth_headers, sample_csv_file):
    """Tests triggering automated training pipeline via API."""
    # Upload sample CSV
    with open(sample_csv_file, "rb") as f:
        upload_res = client.post(
            "/api/v1/dataset/upload",
            files={"file": ("sample.csv", f, "text/csv")},
            headers=auth_headers,
        )
    dataset_id = upload_res.json()["id"]

    # Execute training
    train_payload = {
        "dataset_id": dataset_id,
        "target_column": "target",
        "problem_type": "classification",
        "perform_tuning": False,
    }
    train_res = client.post(
        "/api/v1/train/execute",
        json=train_payload,
        headers=auth_headers,
    )
    assert train_res.status_code == 200
    res_data = train_res.json()
    assert "best_model_name" in res_data
    assert "leaderboard" in res_data


def test_drift_detection_endpoint(client, auth_headers, sample_csv_file):
    """Tests data drift detection endpoint."""
    # Upload dataset as baseline reference
    with open(sample_csv_file, "rb") as f:
        upload_res = client.post(
            "/api/v1/dataset/upload",
            files={"file": ("sample.csv", f, "text/csv")},
            headers=auth_headers,
        )
    dataset_id = upload_res.json()["id"]

    # Construct incoming runtime batch
    current_batch = [
        {"feature_num1": 0.5, "feature_num2": 25.0, "feature_cat": "A"},
        {"feature_num1": 1.2, "feature_num2": 35.0, "feature_cat": "B"},
    ]

    drift_payload = {
        "reference_dataset_id": dataset_id,
        "current_features": current_batch,
    }

    drift_res = client.post(
        "/api/v1/monitor/drift",
        json=drift_payload,
        headers=auth_headers,
    )
    assert drift_res.status_code == 200
    assert "has_drift" in drift_res.json()
    assert "p_values" in drift_res.json()
