#!/bin/bash

# Startup script for Garment Hole Detection API on vast.ai
# This script sets up the environment and starts the API server

set -e

echo "🚀 Starting Garment Hole Detection API setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on vast.ai (look for typical vast.ai environment)
if [ -f "/usr/local/cuda/version.txt" ] || command -v nvidia-smi &> /dev/null; then
    print_status "Detected GPU environment (likely vast.ai)"
    export CUDA_VISIBLE_DEVICES=0
else
    print_warning "No GPU detected, running on CPU"
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export HOST=0.0.0.0
export PORT=8000

# Create directories
print_status "Creating required directories..."
mkdir -p data results debug logs

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found! Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support if GPU is available
if command -v nvidia-smi &> /dev/null; then
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    print_status "Installing PyTorch (CPU only)..."
    pip install torch torchvision
fi

# Install requirements
print_status "Installing requirements..."
pip install -r requirements.txt

# Check if YOLO model exists, download if not
if [ ! -f "yolo11n.pt" ]; then
    print_status "Downloading YOLO model..."
    python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
    print_success "YOLO model downloaded"
else
    print_status "YOLO model already exists"
fi

# Test imports
print_status "Testing critical imports..."
python3 -c "
import cv2
import torch
import transformers
import ultralytics
import fastapi
print('✅ All critical imports successful')
"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    print_status "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA not available, using CPU')
"
fi

# Create a test endpoint script
cat > test_api.py << 'EOF'
#!/usr/bin/env python3
import requests
import time

def test_api():
    base_url = "http://localhost:8000"

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API is healthy")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("Testing API endpoints...")
    test_api()
EOF

chmod +x test_api.py

# Start the API server
print_success "Setup complete! Starting API server..."
print_status "API will be available at: http://0.0.0.0:8000"
print_status "API Documentation: http://0.0.0.0:8000/docs"
print_status "Health Check: http://0.0.0.0:8000/health"

# Start the server with log output
exec python3 api_server.py