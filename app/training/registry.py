"""
Model Registry Mapping and Hyperparameter Grids Definition.
Provides standard classifier and regressor definitions, parameter search spaces,
and evaluator metric mappings.
"""

from typing import Any, Dict, List, Tuple
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class ModelRegistry:
    """Registry mapping candidate algorithms to their estimator instances and default hyperparameter spaces."""

    @staticmethod
    def get_classification_models() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
        """
        Returns a dictionary mapping classifier names to tuples of (Estimator, Hyperparameter Grid).
        """
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, random_state=42),
                {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs", "liblinear"],
                },
            ),
            "Random Forest Classifier": (
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
            ),
            "Decision Tree Classifier": (
                DecisionTreeClassifier(random_state=42),
                {
                    "max_depth": [None, 5, 10, 20],
                    "criterion": ["gini", "entropy"],
                },
            ),
            "Gradient Boosting Classifier": (
                GradientBoostingClassifier(random_state=42),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5],
                },
            ),
            "XGBoost Classifier": (
                xgb.XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6],
                },
            ),
            "LightGBM Classifier": (
                lgb.LGBMClassifier(random_state=42, verbose=-1),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "num_leaves": [15, 31],
                },
            ),
            "CatBoost Classifier": (
                cb.CatBoostClassifier(random_state=42, verbose=0),
                {
                    "iterations": [100, 200],
                    "learning_rate": [0.03, 0.1],
                    "depth": [4, 6],
                },
            ),
            "K-Nearest Neighbors": (
                KNeighborsClassifier(),
                {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
            ),
            "Support Vector Classifier": (
                SVC(probability=True, random_state=42),
                {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["rbf", "linear"],
                },
            ),
            "Gaussian Naive Bayes": (
                GaussianNB(),
                {
                    "var_smoothing": [1e-9, 1e-8, 1e-7],
                },
            ),
            "Extra Trees Classifier": (
                ExtraTreesClassifier(random_state=42),
                {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20],
                },
            ),
        }

    @staticmethod
    def get_regression_models() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
        """
        Returns a dictionary mapping regressor names to tuples of (Estimator, Hyperparameter Grid).
        """
        return {
            "Linear Regression": (
                LinearRegression(),
                {},
            ),
            "Ridge Regression": (
                Ridge(random_state=42),
                {
                    "alpha": [0.1, 1.0, 10.0, 100.0],
                },
            ),
            "Lasso Regression": (
                Lasso(random_state=42),
                {
                    "alpha": [0.01, 0.1, 1.0, 10.0],
                },
            ),
            "Random Forest Regressor": (
                RandomForestRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
            ),
            "Decision Tree Regressor": (
                DecisionTreeRegressor(random_state=42),
                {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5],
                },
            ),
            "Gradient Boosting Regressor": (
                GradientBoostingRegressor(random_state=42),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5],
                },
            ),
            "XGBoost Regressor": (
                xgb.XGBRegressor(random_state=42),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6],
                },
            ),
            "LightGBM Regressor": (
                lgb.LGBMRegressor(random_state=42, verbose=-1),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "num_leaves": [15, 31],
                },
            ),
            "CatBoost Regressor": (
                cb.CatBoostRegressor(random_state=42, verbose=0),
                {
                    "iterations": [100, 200],
                    "learning_rate": [0.03, 0.1],
                    "depth": [4, 6],
                },
            ),
            "K-Nearest Neighbors Regressor": (
                KNeighborsRegressor(),
                {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
            ),
            "Support Vector Regressor": (
                SVR(),
                {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["rbf", "linear"],
                },
            ),
            "Extra Trees Regressor": (
                ExtraTreesRegressor(random_state=42),
                {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20],
                },
            ),
        }
