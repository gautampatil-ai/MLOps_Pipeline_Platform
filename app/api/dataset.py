"""
Dataset Upload & EDA Inspection API Endpoints.
"""

from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.api.auth import get_current_user
from app.models.db_models import Dataset, User
from app.models.schemas import DatasetInfo
from app.preprocessing.eda import EDAEngine

router = APIRouter(prefix="/dataset", tags=["Dataset Management"])


@router.post("/upload", response_model=DatasetInfo, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Uploads CSV or Excel files, reads metadata, and saves dataset to disk."""
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV and Excel files are supported.",
        )

    file_path = settings.RAW_DATA_DIR / file.filename
    contents = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset file: {str(e)}")

    row_count, col_count = df.shape
    dataset_rec = Dataset(
        filename=file.filename,
        filepath=str(file_path),
        row_count=row_count,
        column_count=col_count,
        file_size_bytes=len(contents),
        user_id=current_user.id,
    )
    db.add(dataset_rec)
    db.commit()
    db.refresh(dataset_rec)

    return dataset_rec


@router.get("/{dataset_id}/summary")
def get_dataset_summary(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Extracts profiling and summary statistics for a registered dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset.filepath) if dataset.filename.endswith(".csv") else pd.read_excel(dataset.filepath)
    eda = EDAEngine(df)

    return {
        "dataset_info": DatasetInfo.model_validate(dataset),
        "summary": eda.get_summary_statistics(),
        "missing_report": eda.get_missing_value_report(),
        "outlier_report": eda.get_outlier_report(),
        "correlation": eda.get_correlation_matrix(),
    }
