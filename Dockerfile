# ----------------------
# Stage 1 - Build stage
# ----------------------
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# -------------------------
# Stage 2 - Runtime stage
# -------------------------
FROM python:3.11-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    PORT=10000 \
    FLASK_RUN_PORT=10000 \
    FLASK_RUN_HOST=0.0.0.0 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# ✅ Install libGL (needed for OpenCV/matplotlib/etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# Copy only the virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy project files
COPY . .

# Expose the default Flask port
EXPOSE 10000

# Start the Flask app
CMD ["flask", "run"]
