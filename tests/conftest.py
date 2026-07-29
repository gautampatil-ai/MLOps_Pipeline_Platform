"""
Pytest Fixtures for Database, Test Client, and Sample Data.
"""

import os
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Override Settings Environment Variables before Core Imports
os.environ["DATABASE_URL"] = "sqlite:///./test_mlops.db"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-12345"

from app.main import app
from app.db.session import Base, get_db
from app.core.security import create_access_token, get_password_hash
from app.models.user import User

# In-memory SQLite Database Engine for Testing
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Provides a fresh database session for each test function."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Provides a FastAPI TestClient with database session override."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session):
    """Creates and returns a dummy test user in the database."""
    hashed_pwd = get_password_hash("testpassword123")
    user = User(
        username="testuser",
        email="testuser@example.com",
        hashed_password=hashed_pwd,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Generates valid JWT Authorization headers for test user."""
    token = create_access_token(data={"sub": test_user.username})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_classification_df():
    """Generates a synthetic pandas DataFrame for classification tasks."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_num1": np.random.randn(n),
        "feature_num2": np.random.rand(n) * 100,
        "feature_cat": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })
    # Inject missing values for data cleaning tests
    df.loc[0:5, "feature_num1"] = np.nan
    df.loc[10:12, "feature_cat"] = np.nan
    return df


@pytest.fixture
def sample_csv_file(sample_classification_df):
    """Creates a temporary CSV file on disk for file upload testing."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        sample_classification_df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()
