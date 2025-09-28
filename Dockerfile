# Dockerfile for Garment Hole Detection API
# Optimized for NVIDIA GPU environments like vast.ai

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Download YOLO model if not present
RUN if [ ! -f "yolo11n.pt" ]; then \
    . venv/bin/activate && \
    python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    ; fi

# Create necessary directories
RUN mkdir -p data results debug logs

# Set permissions
RUN chmod +x api_server.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["bash", "-c", ". venv/bin/activate && python3 api_server.py"]