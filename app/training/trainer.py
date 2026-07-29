"""
Automated Model Training, Cross-Validation, Hyperparameter Tuning, and MLflow Tracking Engine.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from app.core.config import settings
from app.core.logger import logger
from app.training.registry import ModelRegistry


class ModelTrainer:
    """
    Orchestrates problem type detection, model iteration, hyperparameter optimization,
    evaluation metric computation, artifact persistence, and MLflow experiment tracking.
    """

    def __init__(
        self,
        target_column: str,
        problem_type: Optional[str] = None,
        test_size: float = settings.TEST_SIZE,
        random_state: int = settings.RANDOM_STATE,
        cv_folds: int = settings.CV_FOLDS,
    ):
        self.target_column = target_column
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds

        # Configure MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    def detect_problem_type(self, y: pd.Series) -> str:
        """
        Determines if the target variable represents a classification or regression task.
        """
        if self.problem_type in ["classification", "regression"]:
            return self.problem_type

        # Heuristic detection based on data type and unique value count
        if y.dtype in ["object", "category", "bool"]:
            return "classification"
        
        unique_count = y.nunique()
        if unique_count <= 20 and (y % 1 == 0).all():
            return "classification"
            
        return "regression"

    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculates classification evaluation metrics."""
        is_binary = len(np.unique(y_true)) <= 2
        average_mode = "binary" if is_binary else "weighted"

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average_mode, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average_mode, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average=average_mode, zero_division=0)),
        }

        if y_proba is not None:
            try:
                if is_binary:
                    # Select positive class probabilities
                    proba_input = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba_input))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC score: {e}")
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.0

        return metrics

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates regression evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
        }

    def train_and_eval_model(
        self,
        name: str,
        estimator: Any,
        param_grid: Dict[str, List[Any]],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str,
        perform_tune: bool = False,
        search_type: str = "grid",
    ) -> Dict[str, Any]:
        """
        Executes cross-validated training, optional hyperparameter search, metrics evaluation,
        and logs parameters and metrics to MLflow.
        """
        logger.info(f"Training pipeline started for model: {name}")

        with mlflow.start_run(run_name=f"{name}_{problem_type}") as run:
            best_params = {}
            fitted_model = estimator

            # Optional Hyperparameter Tuning
            if perform_tune and param_grid:
                scoring = "f1_weighted" if problem_type == "classification" else "neg_mean_squared_error"
                if search_type == "random":
                    search = RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions=param_grid,
                        n_iter=10,
                        cv=self.cv_folds,
                        scoring=scoring,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                else:
                    search = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=self.cv_folds,
                        scoring=scoring,
                        n_jobs=-1,
                    )
                search.fit(X_train, y_train)
                fitted_model = search.best_estimator_
                best_params = search.best_params_
                logger.info(f"Optimal parameters for {name}: {best_params}")
            else:
                fitted_model.fit(X_train, y_train)
                if hasattr(fitted_model, "get_params"):
                    best_params = fitted_model.get_params()

            # Predictions
            y_pred = fitted_model.predict(X_test)
            y_proba = None
            if problem_type == "classification" and hasattr(fitted_model, "predict_proba"):
                try:
                    y_proba = fitted_model.predict_proba(X_test)
                except Exception as e:
                    logger.warning(f"Failed to extract probability predictions for {name}: {e}")

            # Metric Computations
            if problem_type == "classification":
                metrics = self.calculate_classification_metrics(y_test, y_pred, y_proba)
                primary_metric_key = "f1_score"
            else:
                metrics = self.calculate_regression_metrics(y_test, y_pred)
                primary_metric_key = "r2_score"

            # MLflow Logging
            mlflow.log_param("model_name", name)
            mlflow.log_param("problem_type", problem_type)
            for p_name, p_val in best_params.items():
                mlflow.log_param(f"hp_{p_name}", str(p_val))

            for m_name, m_val in metrics.items():
                mlflow.log_metric(m_name, m_val)

            # Persist and Log Model Artifact
            model_path = settings.MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(fitted_model, model_path)
            mlflow.log_artifact(str(model_path))

            return {
                "model_name": name,
                "model": fitted_model,
                "params": best_params,
                "metrics": metrics,
                "primary_score": metrics[primary_metric_key],
                "run_id": run.info.run_id,
                "model_path": str(model_path),
            }

    def train_all_models(
        self,
        df: pd.DataFrame,
        perform_tune: bool = False,
        selected_models: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """
        Splits dataset, identifies candidate models, trains all selected algorithms,
        generates the leaderboard, and identifies the best-performing model.
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in provided dataset.")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Problem Type Inference
        problem_type = self.detect_problem_type(y)
        logger.info(f"Target variable '{self.target_column}' detected as: {problem_type}")

        # Train-Test Split
        stratify_arg = y if problem_type == "classification" and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_arg
        )

        # Retrieve Registry Models
        if problem_type == "classification":
            available_models = ModelRegistry.get_classification_models()
        else:
            available_models = ModelRegistry.get_regression_models()

        if selected_models:
            available_models = {k: v for k, v in available_models.items() if k in selected_models}

        results = []
        model_artifacts = {}

        for name, (estimator, param_grid) in available_models.items():
            try:
                eval_res = self.train_and_eval_model(
                    name=name,
                    estimator=estimator,
                    param_grid=param_grid,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    problem_type=problem_type,
                    perform_tune=perform_tune,
                )
                
                leaderboard_entry = {"Model Name": name, **eval_res["metrics"], "Run ID": eval_res["run_id"]}
                results.append(leaderboard_entry)
                model_artifacts[name] = eval_res
            except Exception as e:
                logger.error(f"Execution failed for model '{name}': {str(e)}", exc_info=True)

        if not results:
            raise RuntimeError("Pipeline failed to successfully train any candidate models.")

        # Leaderboard Generation
        leaderboard_df = pd.DataFrame(results)
        sort_metric = "f1_score" if problem_type == "classification" else "r2_score"
        leaderboard_df = leaderboard_df.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)

        best_model_name = leaderboard_df.iloc[0]["Model Name"]
        best_model_details = model_artifacts[best_model_name]

        # Save Best Model as Global Default Production Model
        prod_model_path = settings.MODELS_DIR / "best_model.joblib"
        joblib.dump(best_model_details["model"], prod_model_path)
        logger.info(f"Top model selected: '{best_model_name}' persisted to {prod_model_path}")

        return leaderboard_df, best_model_details, problem_type
