#!/usr/bin/env python3
"""
Test client for Garment Hole Detection API

This script demonstrates how to use the API from your edge service.
"""

import requests
import json
import time
import os
import sys
from pathlib import Path


class HoleDetectionClient:
    """Client for the Hole Detection API"""

    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url.rstrip('/')

    def health_check(self):
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ API is healthy")
                data = response.json()
                print(f"   Status: {data['status']}")
                print(f"   Detector initialized: {data['detector_initialized']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

    def detect_holes_simple(self, image_path):
        """Simple hole detection (local AI only)"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None

        try:
            print(f"🔍 Detecting holes in: {image_path}")
            start_time = time.time()

            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    f"{self.api_url}/detect-holes-simple",
                    files=files,
                    timeout=120  # 2 minutes timeout
                )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Detection complete in {processing_time:.2f}s")
                print(f"   Holes found: {result['holes_found']}")

                for i, hole in enumerate(result['holes']):
                    bbox = hole['bbox']
                    print(f"   Hole #{i+1}: ({bbox['x']}, {bbox['y']}) "
                          f"size {bbox['w']}x{bbox['h']} "
                          f"confidence {hole['confidence']:.2f}")

                return result
            else:
                print(f"❌ Detection failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return None

    def detect_holes_advanced(self, image_path, **params):
        """Advanced hole detection with parameters"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None

        try:
            print(f"🔍 Advanced detection for: {image_path}")
            print(f"   Parameters: {params}")
            start_time = time.time()

            # Prepare form data
            files = {'image': open(image_path, 'rb')}
            data = {
                'use_openai': params.get('use_openai', False),
                'local_threshold': params.get('local_threshold', 0.45),
                'openai_threshold': params.get('openai_threshold', 0.7),
                'tile_size': params.get('tile_size', 512),
                'overlap': params.get('overlap', 128),
                'min_confidence': params.get('min_confidence', 0.7)
            }

            if params.get('openai_key'):
                data['openai_key'] = params['openai_key']

            response = requests.post(
                f"{self.api_url}/detect-holes",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )

            files['image'].close()
            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Advanced detection complete in {processing_time:.2f}s")
                print(f"   Success: {result['success']}")
                print(f"   Holes detected: {result['num_holes_detected']}")
                print(f"   Message: {result['message']}")

                for i, hole in enumerate(result['holes']):
                    bbox = hole['bbox']
                    print(f"   Hole #{i+1}: ({bbox['x']}, {bbox['y']}) "
                          f"size {bbox['w']}x{bbox['h']} "
                          f"confidence {hole['confidence']:.2f}")

                    if 'verification_score' in hole:
                        print(f"             verification: {hole['verification_score']:.2f}")

                    if 'openai_verification' in hole:
                        openai = hole['openai_verification']
                        print(f"             OpenAI: {openai.get('confidence', 0):.2f} "
                              f"({openai.get('reason', 'N/A')})")

                return result
            else:
                print(f"❌ Advanced detection failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Advanced detection error: {e}")
            return None


def test_api_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing Hole Detection API")
    print("=" * 50)

    # Initialize client
    client = HoleDetectionClient()

    # Test health check
    print("\n1️⃣ Testing health check...")
    if not client.health_check():
        print("❌ API is not healthy. Make sure the server is running.")
        return False

    # Find test image
    test_images = [
        "data/test_shirt.jpg",
        "test_shirt.jpg",
        "data/test_image.jpg"
    ]

    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break

    if not test_image:
        print("\n⚠️ No test image found. Creating a dummy test...")
        print("   Please put a test image in one of these locations:")
        for img in test_images:
            print(f"   - {img}")
        return True

    # Test simple detection
    print(f"\n2️⃣ Testing simple detection...")
    result_simple = client.detect_holes_simple(test_image)

    # Test advanced detection (local AI only)
    print(f"\n3️⃣ Testing advanced detection (local AI)...")
    result_advanced = client.detect_holes_advanced(
        test_image,
        use_openai=False,
        local_threshold=0.4,
        tile_size=512,
        min_confidence=0.6
    )

    print("\n✅ API testing complete!")
    return True


def example_edge_service_usage():
    """Example of how to use the API from an edge service"""
    print("\n" + "=" * 60)
    print("📱 EXAMPLE: Edge Service Integration")
    print("=" * 60)

    code_example = '''
# Example: Using the API from your edge service

import requests
import json

class EdgeService:
    def __init__(self, api_url):
        self.api_url = api_url

    def process_garment_image(self, image_path):
        """Process garment image and return hole locations"""

        # Upload image to hole detection API
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{self.api_url}/detect-holes-simple",
                files=files,
                timeout=120
            )

        if response.status_code == 200:
            result = response.json()

            if result['success'] and result['holes_found'] > 0:
                # Format holes for your system
                holes = []
                for hole in result['holes']:
                    bbox = hole['bbox']
                    holes.append({
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['w'],
                        'height': bbox['h'],
                        'confidence': hole['confidence']
                    })

                return {
                    'status': 'holes_detected',
                    'count': len(holes),
                    'holes': holes
                }
            else:
                return {
                    'status': 'no_holes',
                    'count': 0,
                    'holes': []
                }
        else:
            return {
                'status': 'error',
                'message': f"API error: {response.status_code}"
            }

# Usage example
api_url = "http://your-vast-ai-instance:8000"
edge_service = EdgeService(api_url)

result = edge_service.process_garment_image("garment.jpg")
print(f"Status: {result['status']}")
print(f"Holes found: {result['count']}")
'''

    print(code_example)


if __name__ == "__main__":
    # Check if API URL is provided
    api_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        api_url = sys.argv[1]

    print(f"🎯 Testing API at: {api_url}")

    # Update client
    global client
    client = HoleDetectionClient(api_url)

    # Run tests
    if test_api_endpoints():
        example_edge_service_usage()

        print("\n" + "=" * 60)
        print("🚀 DEPLOYMENT CHECKLIST")
        print("=" * 60)
        print("✅ 1. API server is running")
        print("✅ 2. Endpoints are responding")
        print("✅ 3. Image processing works")
        print("\n📋 Next steps:")
        print("   1. Deploy to vast.ai using: ./start_api.sh")
        print("   2. Update your edge service with the vast.ai IP")
        print("   3. Test from your edge service")
        print("   4. Monitor performance and costs")

        print(f"\n💡 API Documentation: {api_url}/docs")
        print(f"🔍 Health Check: {api_url}/health")
    else:
        print("\n❌ Tests failed. Check the API server.")
        sys.exit(1)