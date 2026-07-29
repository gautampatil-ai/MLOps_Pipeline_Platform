"""
Central REST API Router combining all domain sub-routers into a unified v1 namespace.
"""

from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.dataset import router as dataset_router
from app.api.train import router as train_router
from app.api.predict import router as predict_router
from app.api.monitor import router as monitor_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(auth_router)
api_v1_router.include_router(dataset_router)
api_v1_router.include_router(train_router)
api_v1_router.include_router(predict_router)
api_v1_router.include_router(monitor_router)
