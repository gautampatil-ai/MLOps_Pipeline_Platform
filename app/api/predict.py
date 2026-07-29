"""
Inference API Endpoint for serving single-instance real-time model predictions.
"""

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post("/", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Consumes input payload dict, loads production best model artifact, and returns inference.
    """
    model_path = settings.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No production model found. Please run a training pipeline first.",
        )

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model artifact: {str(e)}")

    input_df = pd.DataFrame([req.features])

    try:
        prediction_val = model.predict(input_df)[0]
        # Cast numpy types to native python types
        if hasattr(prediction_val, "item"):
            prediction_val = prediction_val.item()

        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                proba_arr = model.predict_proba(input_df)[0]
                classes = getattr(model, "classes_", range(len(proba_arr)))
                probabilities = {str(c): float(p) for c, p in zip(classes, proba_arr)}
            except Exception:
                probabilities = None

        return PredictionResponse(
            prediction=prediction_val,
            probabilities=probabilities,
            model_version="best_model.joblib",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")
