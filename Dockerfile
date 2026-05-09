FROM python:3.11-slim

WORKDIR /app

# LightGBM needs these system libs
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/         ./app/
COPY src/         ./src/
COPY scania_pipeline.py  ./
# COPY scania_labels.py    ./

# Copy trained model artifacts
# These are produced by run_experiments.py and must exist before building
COPY outputs/models/pipeline_state.pkl      ./outputs/models/pipeline_state.pkl
COPY outputs/models/exp1_log_loss.pkl       ./outputs/models/exp1_log_loss.pkl
COPY outputs/models/exp2_focal_loss.pkl     ./outputs/models/exp2_focal_loss.pkl
COPY outputs/models/exp3_cost_aware_focal_fn1_0.pkl ./outputs/models/exp3_cost_aware_focal_fn1_0.pkl

ENV PORT=8080
ENV OUTPUTS_DIR=outputs/models
ENV ACTIVE_MODEL=exp3
ENV MODEL_VERSION=1.0.0

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]