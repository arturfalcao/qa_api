#!/usr/bin/env python3
"""
Garment Hole Detection System - Main Entry Point

A comprehensive hole detection system using multi-model AI approach:
- Segmented detection with tiling
- Local AI filtering (YOLO + ResNet-18 + hand-crafted features)
- OpenAI Vision API verification

Usage:
    python main.py --image path/to/image.jpg --api-key YOUR_OPENAI_KEY
    python main.py --test  # Run with built-in test image
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from integrated_hole_pipeline import IntegratedHolePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Garment Hole Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --test --api-key sk-...
  python main.py --image data/test_shirt.jpg --api-key sk-...
  python main.py --image my_image.jpg --api-key sk-... --local-only
        """
    )

    parser.add_argument(
        "--image",
        type=str,
        default="data/test_shirt.jpg",
        help="Path to input image (default: data/test_shirt.jpg)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key for final verification"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test with built-in test image"
    )

    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run local AI filtering only (no OpenAI verification)"
    )

    parser.add_argument(
        "--local-threshold",
        type=float,
        default=0.45,
        help="Local AI filter threshold (default: 0.45)"
    )

    parser.add_argument(
        "--openai-threshold",
        type=float,
        default=0.7,
        help="OpenAI verification threshold (default: 0.7)"
    )

    parser.add_argument(
        "--max-openai-calls",
        type=int,
        default=15,
        help="Maximum OpenAI API calls for cost control (default: 15)"
    )

    parser.add_argument(
        "--advanced-ai",
        action="store_true",
        help="Use advanced local AI models (RTX 5090 optimized)"
    )

    parser.add_argument(
        "--hardware-config",
        action="store_true",
        help="Analyze and display hardware configuration"
    )

    args = parser.parse_args()

    # Handle hardware config display
    if args.hardware_config:
        from hardware_config import RTX5090Optimizer
        optimizer = RTX5090Optimizer()
        optimizer.print_optimization_summary()
        return 0

    # Validate inputs
    if not args.local_only and not args.api_key:
        print("❌ Error: OpenAI API key required unless using --local-only")
        parser.print_help()
        return 1

    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return 1

    # Initialize pipeline based on mode
    if args.advanced_ai:
        print("🚀 Using advanced AI mode (RTX 5090 optimized)")
        from advanced_local_ai_filter import AdvancedLocalAIFilter
        # Use advanced filter instead of regular one
        api_key = args.api_key or "local-only"
        pipeline = IntegratedHolePipeline(api_key)
        # Replace local filter with advanced version
        pipeline.local_filter = AdvancedLocalAIFilter()
    else:
        api_key = args.api_key or "local-only"
        pipeline = IntegratedHolePipeline(api_key)

    print(f"🎯 Running hole detection on: {args.image}")

    if args.local_only:
        # Local AI filtering only
        print("🔧 Mode: Local AI filtering only")
        import cv2
        import json

        # Load image and run local detection
        image = cv2.imread(args.image)

        # Use enhanced detections if available, otherwise need to run full detection
        enhanced_detections_path = "results/enhanced_detections.json"
        if os.path.exists(enhanced_detections_path):
            with open(enhanced_detections_path, 'r') as f:
                enhanced_detections = json.load(f)
        else:
            print("❌ Enhanced detections not found. Run full pipeline first.")
            return 1

        # Apply local filtering
        local_filtered = pipeline.apply_local_filter(
            image, enhanced_detections, threshold=args.local_threshold
        )

        print(f"✅ Local AI filtering complete: {len(enhanced_detections)} → {len(local_filtered)} detections")

    else:
        # Full pipeline with OpenAI verification
        print("🔧 Mode: Full pipeline with OpenAI verification")
        final_detections = pipeline.run_complete_pipeline(
            image_path=args.image,
            enhanced_detections_path="results/enhanced_detections.json",
            local_threshold=args.local_threshold,
            openai_threshold=args.openai_threshold,
            max_openai_calls=args.max_openai_calls
        )

        print(f"✅ Pipeline complete: {len(final_detections)} verified hole(s) detected")

    return 0


if __name__ == "__main__":
    sys.exit(main())