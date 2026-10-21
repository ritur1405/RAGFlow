FROM python:3.11-slim

WORKDIR /app

# psycopg2 and pgvector need a compiler and libpq headers at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so dependency layers are cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the cross-encoder reranker into the image so containers start without a
# download and work offline. Keep RERANKER_MODEL in step with app/config.py.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1
ARG RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
ENV RERANKER_MODEL=${RERANKER_MODEL}
RUN python -c "import os; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name=os.environ['RERANKER_MODEL'])"

COPY . .

EXPOSE 8000

# Table creation happens in the app's lifespan hook, not at import time, so the
# container can start before the database finishes accepting connections.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
