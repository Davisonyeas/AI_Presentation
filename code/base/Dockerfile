FROM python:3.11-slim

ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \ 
    gradio>=4.44.0 \
    transformers>=4.44.0 \
    torch>=2.4.0

WORKDIR /app

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
