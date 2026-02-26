# Multi-stage build for anomaly detection system
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libsasl2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data/raw data/processed models logs

# Production stage
FROM base as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "src/train.py", "--help"]


# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest \
    pytest-cov \
    black \
    flake8

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
