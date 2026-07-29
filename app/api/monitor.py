"""
Model Performance & Data Drift Monitoring API Endpoints.
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import Dataset
from app.models.schemas import DriftRequest, DriftResponse
from app.monitoring.drift_detector import DataDriftDetector

router = APIRouter(prefix="/monitor", tags=["Monitoring"])


@router.post("/drift", response_model=DriftResponse)
def check_data_drift(req: DriftRequest, db: Session = Depends(get_db)):
    """
    Compares uploaded baseline reference dataset against incoming current features
    using Kolmogorov-Smirnov / Chi-Square tests to detect statistical drift.
    """
    reference = db.query(Dataset).filter(Dataset.id == req.reference_dataset_id).first()
    if not reference:
        raise HTTPException(status_code=404, detail="Baseline reference dataset not found.")

    ref_df = pd.read_csv(reference.filepath) if reference.filename.endswith(".csv") else pd.read_excel(reference.filepath)
    curr_df = pd.DataFrame(req.current_features)

    detector = DataDriftDetector(reference_data=ref_df)
    drift_result = detector.detect_drift(current_data=curr_df)

    return DriftResponse(
        has_drift=drift_result["has_drift"],
        drifted_features=drift_result["drifted_features"],
        p_values=drift_result["p_values"],
        details=drift_result["details"],
    )
