import os
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application Core Settings loaded from YAML and Environment Variables.
    """

    APP_NAME: str = "MLOps Production Pipeline Platform"
    APP_VERSION: str = "1.0.0"
    ENV: str = "production"
    SECRET_KEY: str = "SUPER_SECRET_KEY_CHANGE_IN_PRODUCTION_32_BYTES"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DATABASE_URL: str = "sqlite:///./data/mlops_platform.db"

    RAW_DATA_DIR: Path = Path("./data/raw")
    PROCESSED_DATA_DIR: Path = Path("./data/processed")
    MODELS_DIR: Path = Path("./models")
    REPORTS_DIR: Path = Path("./reports")
    LOGS_DIR: Path = Path("./logs")

    MLFLOW_TRACKING_URI: str = "sqlite:///./mlruns/mlflow.db"
    MLFLOW_ARTIFACT_LOCATION: str = "./mlruns/artifacts"
    MLFLOW_EXPERIMENT_NAME: str = "MLOps_Automated_Pipeline"

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5

    DRIFT_THRESHOLD_P_VALUE: float = 0.05

    class Config:
        env_file = ".env"
        extra = "ignore"


def load_yaml_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Helper function to parse config.yaml if present on disk."""
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# Initialize global settings
yaml_cfg = load_yaml_config()

settings = Settings(
    APP_NAME=yaml_cfg.get("app", {}).get("name", "MLOps Platform"),
    APP_VERSION=yaml_cfg.get("app", {}).get("version", "1.0.0"),
    ENV=yaml_cfg.get("app", {}).get("env", "production"),
    SECRET_KEY=yaml_cfg.get("app", {}).get("secret_key", "SUPER_SECRET_KEY"),
    ALGORITHM=yaml_cfg.get("app", {}).get("algorithm", "HS256"),
    ACCESS_TOKEN_EXPIRE_MINUTES=yaml_cfg.get("app", {}).get(
        "access_token_expire_minutes", 120
    ),
    HOST=yaml_cfg.get("server", {}).get("host", "0.0.0.0"),
    PORT=yaml_cfg.get("server", {}).get("port", 8000),
    DATABASE_URL=yaml_cfg.get("database", {}).get(
        "url", "sqlite:///./data/mlops_platform.db"
    ),
    RAW_DATA_DIR=Path(yaml_cfg.get("paths", {}).get("raw_data_dir", "./data/raw")),
    PROCESSED_DATA_DIR=Path(
        yaml_cfg.get("paths", {}).get("processed_data_dir", "./data/processed")
    ),
    MODELS_DIR=Path(yaml_cfg.get("paths", {}).get("models_dir", "./models")),
    REPORTS_DIR=Path(yaml_cfg.get("paths", {}).get("reports_dir", "./reports")),
    LOGS_DIR=Path(yaml_cfg.get("paths", {}).get("logs_dir", "./logs")),
    MLFLOW_TRACKING_URI=yaml_cfg.get("mlflow", {}).get(
        "tracking_uri", "sqlite:///./mlruns/mlflow.db"
    ),
    MLFLOW_ARTIFACT_LOCATION=yaml_cfg.get("mlflow", {}).get(
        "artifact_location", "./mlruns/artifacts"
    ),
    MLFLOW_EXPERIMENT_NAME=yaml_cfg.get("mlflow", {}).get(
        "experiment_name", "MLOps_Automated_Pipeline"
    ),
    TEST_SIZE=yaml_cfg.get("training", {}).get("test_size", 0.2),
    RANDOM_STATE=yaml_cfg.get("training", {}).get("random_state", 42),
    CV_FOLDS=yaml_cfg.get("training", {}).get("cv_folds", 5),
    DRIFT_THRESHOLD_P_VALUE=yaml_cfg.get("monitoring", {}).get(
        "drift_threshold_p_value", 0.05
    ),
)

# Ensure directories exist upon import
for path in [
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.MODELS_DIR,
    settings.REPORTS_DIR,
    settings.LOGS_DIR,
    Path("./mlruns"),
]:
    path.mkdir(parents=True, exist_ok=True)
