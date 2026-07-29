# Multi-stage production Dockerfile for MLOps FastAPI & Streamlit Service
FROM python:3.11-slim as builder

WORKDIR /app

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies manifest and install wheel packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final Runtime Stage
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# Install C libraries required by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy Application Source Code
COPY . /app

# Create persistent storage directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/mlruns

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
