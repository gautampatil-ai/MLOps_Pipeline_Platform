"""
Automated Training & Experiment Orchestration Endpoints.
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.auth import get_current_user
from app.models.db_models import Dataset, ExperimentRun, User
from app.models.schemas import TrainRequest, TrainResponse
from app.preprocessing.cleaner import DataCleaner
from app.training.trainer import ModelTrainer

router = APIRouter(prefix="/train", tags=["Training Pipeline"])


@router.post("/execute", response_model=TrainResponse)
def execute_training_pipeline(
    req: TrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Executes end-to-end dataset cleaning, model training across algorithms,
    leaderboard generation, and MLflow run logging.
    """
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Requested dataset does not exist.")

    # Load raw dataset
    df = pd.read_csv(dataset.filepath) if dataset.filename.endswith(".csv") else pd.read_excel(dataset.filepath)

    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' missing from dataset.")

    # Clean non-target features
    X_raw = df.drop(columns=[req.target_column])
    y_raw = df[req.target_column]

    cleaner = DataCleaner()
    X_cleaned = cleaner.fit_transform(X_raw)
    cleaned_df = pd.concat([X_cleaned, y_raw.reset_index(drop=True)], axis=1)

    # Initialize Trainer and execute models
    trainer = ModelTrainer(
        target_column=req.target_column,
        problem_type=req.problem_type,
    )

    try:
        leaderboard_df, best_details, problem_type = trainer.train_all_models(
            df=cleaned_df,
            perform_tune=req.perform_tuning,
            selected_models=req.selected_models,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training pipeline error: {str(e)}")

    # Persist Experiment Record in SQLite DB
    exp_record = ExperimentRun(
        experiment_name=f"Exp_{dataset.filename}_{req.target_column}",
        problem_type=problem_type,
        target_column=req.target_column,
        best_model_name=best_details["model_name"],
        best_score=best_details["primary_score"],
        mlflow_run_id=best_details["run_id"],
        leaderboard_json=leaderboard_df.to_dict(orient="records"),
        user_id=current_user.id,
    )
    db.add(exp_record)
    db.commit()
    db.refresh(exp_record)

    return TrainResponse(
        experiment_id=exp_record.id,
        problem_type=problem_type,
        best_model_name=best_details["model_name"],
        best_score=best_details["primary_score"],
        mlflow_run_id=best_details["run_id"],
        leaderboard=leaderboard_df.to_dict(orient="records"),
    )
