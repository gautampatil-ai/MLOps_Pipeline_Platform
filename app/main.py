"""
Main FastAPI Application Entrypoint & Middleware Configuration.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.endpoints import auth, dataset, train, inference, explain, monitor
from app.core.config import settings
from app.db.session import engine, Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown routines."""
    # Ensure database tables exist on startup
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Prometheus Metrics Instrumentation
Instrumentator().instrument(app).expose(app)

# Configure Cross-Origin Resource Sharing (CORS)
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register V1 API Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(dataset.router, prefix=f"{settings.API_V1_STR}/dataset", tags=["Datasets"])
app.include_router(train.router, prefix=f"{settings.API_V1_STR}/train", tags=["Training"])
app.include_router(inference.router, prefix=f"{settings.API_V1_STR}/inference", tags=["Inference"])
app.include_router(explain.router, prefix=f"{settings.API_V1_STR}/explain", tags=["Explainability"])
app.include_router(monitor.router, prefix=f"{settings.API_V1_STR}/monitor", tags=["Monitoring"])


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "ok", "app": settings.PROJECT_NAME}
