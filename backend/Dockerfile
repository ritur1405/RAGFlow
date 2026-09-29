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

COPY . .

EXPOSE 8000

# Table creation happens in the app's lifespan hook, not at import time, so the
# container can start before the database finishes accepting connections.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
