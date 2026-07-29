"""
Unit Tests for Model Trainer & Auto-ML Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from app.training.trainer import ModelTrainer


def test_trainer_classification_pipeline(sample_classification_df):
    """Tests automated model selection and evaluation for classification tasks."""
    trainer = ModelTrainer(
        df=sample_classification_df,
        target_column="target",
        problem_type="classification",
        perform_tuning=False,
    )

    results = trainer.train_all()

    assert "best_model" in results
    assert "best_model_name" in results
    assert "leaderboard" in results
    assert len(results["leaderboard"]) > 0

    # Ensure leaderboard contains accuracy and f1_score metrics
    top_entry = results["leaderboard"][0]
    assert "accuracy" in top_entry or "f1_score" in top_entry or "Model Name" in top_entry


def test_trainer_regression_pipeline():
    """Tests automated model selection and evaluation for regression tasks."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n) * 5,
        "target": np.random.randn(n) * 10,
    })

    trainer = ModelTrainer(
        df=df,
        target_column="target",
        problem_type="regression",
        perform_tuning=False,
    )

    results = trainer.train_all()

    assert results["best_model_name"] is not None
    assert len(results["leaderboard"]) > 0
    top_entry = results["leaderboard"][0]
    assert "rmse" in top_entry or "r2_score" in top_entry or "Model Name" in top_entry


def test_trainer_invalid_target_raises_error(sample_classification_df):
    """Tests that passing a non-existent target column raises ValueError."""
    with pytest.raises(ValueError):
        trainer = ModelTrainer(
            df=sample_classification_df,
            target_column="non_existent_column",
            problem_type="classification",
        )
        trainer.train_all()
