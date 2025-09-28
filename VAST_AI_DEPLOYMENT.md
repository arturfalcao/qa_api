# Garment Hole Detection API - Vast.ai Deployment Guide

This guide explains how to deploy the Garment Hole Detection API on vast.ai.

## 🎯 Quick Start

### Option 1: Direct Setup (Recommended)

1. **Create vast.ai instance with these specs:**
   - Template: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
   - GPU: Any with >= 8GB VRAM (RTX 3080, RTX 4090, A5000, etc.)
   - Disk: >= 30GB
   - RAM: >= 16GB

2. **Connect via SSH and setup:**
   ```bash
   # Clone/upload your project files to the instance
   git clone <your-repo-url> /workspace/hole-detection
   cd /workspace/hole-detection

   # OR upload via SCP:
   # scp -r . root@<instance-ip>:/workspace/hole-detection

   # Run setup script
   chmod +x start_api.sh
   ./start_api.sh
   ```

3. **API will be available at:**
   - Health: `http://<instance-ip>:8000/health`
   - Docs: `http://<instance-ip>:8000/docs`
   - API: `http://<instance-ip>:8000/detect-holes`

### Option 2: Docker Deployment

1. **Build and run with Docker:**
   ```bash
   # Build image
   docker build -t hole-detection-api .

   # Run container
   docker run -d \
     --gpus all \
     -p 8000:8000 \
     --name hole-detection \
     hole-detection-api
   ```

## 🔧 Configuration

### Environment Variables

Set these in your startup command or Docker:

```bash
export HOST=0.0.0.0                    # Listen on all interfaces
export PORT=8000                       # API port
export CUDA_VISIBLE_DEVICES=0          # Use first GPU
export OPENAI_API_KEY=sk-...           # Optional: for OpenAI verification
```

### vast.ai Specific Configuration

1. **Port Forwarding:**
   - vast.ai automatically exposes ports
   - Your API will be available at: `http://<instance-ip>:8000`

2. **GPU Setup:**
   - The instance should have CUDA pre-installed
   - Verify with: `nvidia-smi`

3. **Persistent Storage:**
   - Use `/workspace` for persistent files
   - Models and results will be saved here

## 📊 Instance Specifications

### Minimum Requirements:
- **GPU**: 6GB VRAM (RTX 3060, GTX 1660 Ti)
- **RAM**: 12GB
- **Storage**: 25GB
- **CUDA**: 11.8+

### Recommended (for best performance):
- **GPU**: 12GB+ VRAM (RTX 4070 Ti, RTX 3080, A5000)
- **RAM**: 32GB+
- **Storage**: 50GB+
- **CUDA**: 12.1+

### Your Current Instance:
```
Instance ID: 26288285
Host: 73118
Machine ID: 11742
VRAM: 63.7 GB (Excellent! Can handle large images)
CPU: AMD EPYC 7532 32-Core
RAM: 1026.0 GB (Perfect for batch processing)
```

## 🚀 API Usage Examples

### Health Check
```bash
curl http://<instance-ip>:8000/health
```

### Simple Hole Detection
```bash
curl -X POST \
  http://<instance-ip>:8000/detect-holes-simple \
  -F "image=@your_image.jpg"
```

### Advanced Detection with Parameters
```bash
curl -X POST \
  http://<instance-ip>:8000/detect-holes \
  -F "image=@your_image.jpg" \
  -F "use_openai=false" \
  -F "local_threshold=0.45" \
  -F "tile_size=512" \
  -F "min_confidence=0.7"
```

### From Your Edge Service (Python)
```python
import requests

def detect_holes(image_path, api_url):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{api_url}/detect-holes-simple", files=files)

    return response.json()

# Usage
api_url = "http://<instance-ip>:8000"
result = detect_holes("garment.jpg", api_url)
print(f"Found {result['holes_found']} holes")
for hole in result['holes']:
    bbox = hole['bbox']
    print(f"Hole at ({bbox['x']}, {bbox['y']}) size {bbox['w']}x{bbox['h']}")
```

## 📈 Performance Optimization

### For High-Throughput:

1. **Batch Processing:**
   ```python
   # Process multiple images in parallel
   import asyncio
   import aiohttp

   async def process_images(image_paths, api_url):
       async with aiohttp.ClientSession() as session:
           tasks = []
           for path in image_paths:
               task = upload_image(session, path, api_url)
               tasks.append(task)
           results = await asyncio.gather(*tasks)
       return results
   ```

2. **GPU Memory Management:**
   ```bash
   # Monitor GPU usage
   watch -n 1 nvidia-smi

   # If needed, restart API to clear VRAM
   docker restart hole-detection
   ```

### For Cost Optimization:

1. **Use Local AI Only:**
   - Set `use_openai=false` to avoid OpenAI API costs
   - Local AI is fast and accurate for most cases

2. **Adjust Detection Parameters:**
   ```python
   # Faster but less sensitive
   params = {
       "tile_size": 256,        # Smaller tiles = faster
       "local_threshold": 0.6,  # Higher threshold = fewer detections
       "min_confidence": 0.8    # Higher confidence = fewer false positives
   }
   ```

## 🔍 Monitoring & Debugging

### Log Files:
```bash
# API logs (if using start_api.sh)
tail -f logs/api.log

# Docker logs (if using Docker)
docker logs -f hole-detection
```

### Test Script:
```bash
# Run API health test
python3 test_api.py

# Test with sample image
curl -X POST \
  http://localhost:8000/detect-holes-simple \
  -F "image=@data/test_shirt.jpg"
```

### GPU Monitoring:
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🛠️ Troubleshooting

### Common Issues:

1. **CUDA Out of Memory:**
   ```bash
   # Reduce tile size
   tile_size = 256  # instead of 512

   # Or restart the API
   docker restart hole-detection
   ```

2. **API Not Responding:**
   ```bash
   # Check if port is open
   netstat -tulpn | grep 8000

   # Check firewall (if needed)
   ufw allow 8000
   ```

3. **Model Loading Errors:**
   ```bash
   # Re-download models
   rm -f yolo11n.pt
   python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
   ```

### Performance Issues:

1. **Slow Detection:**
   - Reduce `tile_size` to 256 or 384
   - Increase `min_confidence` threshold
   - Use `detect-holes-simple` endpoint

2. **High Memory Usage:**
   - Process images one at a time
   - Restart API periodically: `docker restart hole-detection`

## 💰 Cost Estimation

### vast.ai Costs (approximate):
- **RTX 3080**: $0.20-0.40/hour
- **RTX 4090**: $0.40-0.80/hour
- **A5000**: $0.30-0.60/hour

### API Costs:
- **Local AI only**: No additional costs
- **With OpenAI**: ~$0.01 per image verification

### Example Monthly Cost:
- Instance: RTX 4090 @ $0.60/hour × 24/7 = ~$432/month
- Processing: 10,000 images/month local = $0
- Total: ~$432/month for 24/7 availability

## 🔒 Security Notes

1. **Firewall**: vast.ai instances are internet-accessible
2. **API Keys**: Store OpenAI keys securely, don't commit to git
3. **Access Control**: Consider adding authentication for production
4. **Data Privacy**: Images are processed locally, not sent to external services (unless using OpenAI)

## 📞 Support

- Check API docs: `http://<instance-ip>:8000/docs`
- Monitor logs: `docker logs hole-detection`
- GPU status: `nvidia-smi`
- Instance terminal: SSH to vast.ai instance

## 🎉 You're Ready!

Your hole detection API is now running on vast.ai and ready to process garment images from your edge service!

Test it with:
```bash
curl http://<your-instance-ip>:8000/health
```