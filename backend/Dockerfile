FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-cloud-storage google-cloud-aiplatform

COPY app/               ./app/
COPY src/               ./src/
COPY scania_pipeline.py ./
COPY scania_labels.py   ./

ENV PORT=8080
ENV GCP_PROJECT_ID=predictive-maintenance-scania
ENV GCS_BUCKET=predictive-maintenance-scania-artifacts
ENV OUTPUTS_DIR=outputs/models
ENV ACTIVE_MODEL=exp1
ENV MODEL_VERSION=1.0.0

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"]
