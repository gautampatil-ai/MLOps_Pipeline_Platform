"""
SQLAlchemy ORM Database Schema Definitions.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    """User account entity for dashboard access and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    datasets = relationship("Dataset", back_populates="owner")
    experiments = relationship("ExperimentRun", back_populates="owner")


class Dataset(Base):
    """Metadata tracking for uploaded datasets."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="datasets")


class ExperimentRun(Base):
    """Metadata tracking for executed pipeline model training runs."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(255), nullable=False)
    problem_type = Column(String(50), nullable=False)
    target_column = Column(String(100), nullable=False)
    best_model_name = Column(String(100), nullable=False)
    best_score = Column(Float, nullable=False)
    mlflow_run_id = Column(String(100), nullable=False)
    leaderboard_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="experiments")
