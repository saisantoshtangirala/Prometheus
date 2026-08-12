FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="Prometheus AI"
LABEL description="Prometheus – The Causal Brain: Market Intelligence System"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /prometheus

# Install Python dependencies in layers (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch>=2.1.0 \
        numpy>=1.24.0 \
        scipy>=1.11.0 \
        scikit-learn>=1.3.0 \
        networkx>=3.1 \
        pandas>=2.0.0 \
        plotly>=5.17.0 \
        loguru>=0.7.0 \
        rich>=13.6.0 \
        python-dotenv>=1.0.0 \
        pydantic>=2.4.0 \
        fastapi>=0.104.0 \
        uvicorn>=0.24.0 \
        yfinance>=0.2.31

# Optional heavy deps (comment out if running CPU-only)
RUN pip install --no-cache-dir \
        dowhy>=0.11.0 \
        diffusers>=0.24.0 \
        transformers>=4.35.0 \
    || echo "Optional deps skipped"

# Copy source
COPY . .

# Create runtime directories
RUN mkdir -p output checkpoints data/cache data/scenarios logs

# Environment
ENV PYTHONPATH=/prometheus
ENV TORCH_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV PROMETHEUS_DEVICE=cpu

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from prometheus.engine import PrometheusEngine; print('OK')" || exit 1

EXPOSE 8000

# Default: interactive Python shell with Prometheus imported
CMD ["python", "-c", "\
import logging; \
logging.basicConfig(level=logging.INFO); \
from prometheus.engine import PrometheusEngine, PrometheusConfig; \
cfg = PrometheusConfig(n_assets=10, seq_len=32, horizon=3); \
engine = PrometheusEngine(cfg); \
print('Prometheus Engine ready. Use engine.analyze() or scripts/train.py'); \
"]
