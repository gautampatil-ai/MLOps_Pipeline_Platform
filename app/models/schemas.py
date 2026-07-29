"""
Pydantic Request, Response, and Data Transfer Object (DTO) Schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field


# --- Auth Schemas ---
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Dataset Schemas ---
class DatasetInfo(BaseModel):
    id: int
    filename: str
    filepath: str
    row_count: int
    column_count: int
    file_size_bytes: int
    created_at: datetime

    class Config:
        from_attributes = True


# --- Training Schemas ---
class TrainRequest(BaseModel):
    dataset_id: int
    target_column: str
    problem_type: Optional[str] = None
    perform_tuning: bool = False
    selected_models: Optional[List[str]] = None


class TrainResponse(BaseModel):
    experiment_id: int
    problem_type: str
    best_model_name: str
    best_score: float
    mlflow_run_id: str
    leaderboard: List[Dict[str, Any]]


# --- Predict Schemas ---
class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: Any
    probabilities: Optional[Dict[str, float]] = None
    model_version: str = "best_model"


# --- Monitoring Schemas ---
class DriftRequest(BaseModel):
    reference_dataset_id: int
    current_features: List[Dict[str, Any]]


class DriftResponse(BaseModel):
    has_drift: bool
    drifted_features: List[str]
    p_values: Dict[str, float]
    details: Dict[str, Any]
