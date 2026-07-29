"""
Database Connection & Session Management Setup.
Configures SQLAlchemy Engine and Scoped Session creation.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.core.config import settings

# SQLite require check_same_thread=False for multi-threaded FastAPI execution
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session for a web request and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
