#!/usr/bin/env python3
"""
FastAPI server for Garment Hole Detection System

Provides REST API endpoints for image upload and hole detection.
Returns JSON response with bounding boxes of detected holes.
"""

import os
import sys
import cv2
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from integrated_hole_pipeline import IntegratedHolePipeline
from verify_holes_ai import VerifiedHoleDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Garment Hole Detection API",
    description="AI-powered hole detection system for garment images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance (initialized lazily)
detector = None
pipeline = None

class DetectionResponse(BaseModel):
    """Response model for hole detection"""
    success: bool
    num_holes_detected: int
    holes: List[Dict]
    message: str
    processing_time_seconds: float

class HoleDetection(BaseModel):
    """Model for individual hole detection"""
    bbox: Dict[str, int]  # x, y, w, h
    confidence: float
    verification_score: Optional[float] = None
    area_pixels: float

def initialize_detector(use_openai: bool = False, openai_key: Optional[str] = None):
    """Initialize the hole detection system"""
    global detector, pipeline

    if detector is None:
        logger.info("Initializing hole detection system...")

        if use_openai and openai_key:
            logger.info("Using integrated pipeline with OpenAI verification")
            pipeline = IntegratedHolePipeline(openai_key)
        else:
            logger.info("Using local AI verification only")
            detector = VerifiedHoleDetector(use_ai_verification=True)

        logger.info("Hole detection system initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Garment Hole Detection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "detector_initialized": detector is not None or pipeline is not None,
        "api_version": "1.0.0"
    }

@app.post("/detect-holes", response_model=DetectionResponse)
async def detect_holes(
    image: UploadFile = File(..., description="Image file to analyze for holes"),
    use_openai: bool = Form(False, description="Use OpenAI verification (requires API key)"),
    openai_key: Optional[str] = Form(None, description="OpenAI API key"),
    local_threshold: float = Form(0.45, description="Local AI filter threshold"),
    openai_threshold: float = Form(0.7, description="OpenAI verification threshold"),
    tile_size: int = Form(512, description="Tile size for segmented detection"),
    overlap: int = Form(128, description="Tile overlap for segmented detection"),
    min_confidence: float = Form(0.7, description="Minimum detection confidence"),
    advanced_ai: bool = Form(False, description="Use advanced RTX 5090 optimized AI models"),
    fabric_optimized: bool = Form(False, description="Use fabric-optimized models for maximum defect detection"),
    winclip: bool = Form(False, description="Use WinCLIP zero-shot anomaly detection (arXiv:2303.14814)"),
    zero_shot_pipeline: bool = Form(False, description="Use complete zero-shot pipeline: WinCLIP + SAM2 + Florence-2 + PatchCore"),
    simplified_zero_shot: bool = Form(False, description="Use simplified zero-shot pipeline (stable version)")
):
    """
    Detect holes in uploaded garment image

    Returns JSON with bounding boxes of detected holes
    """
    import time
    start_time = time.time()

    try:
        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {image.content_type}. Allowed: {allowed_types}"
            )

        # Initialize detector if needed
        initialize_detector(use_openai=use_openai, openai_key=openai_key)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Verify image can be loaded
            test_image = cv2.imread(temp_file_path)
            if test_image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            logger.info(f"Processing image: {image.filename} ({test_image.shape})")

            # Run hole detection
            if use_openai and pipeline:
                # Use integrated pipeline with OpenAI
                logger.info("Running integrated pipeline with OpenAI verification")

                # First need to run initial detection to get enhanced_detections
                from detect_holes_segmented import SegmentedHoleDetector
                initial_detector = SegmentedHoleDetector()
                initial_detections = initial_detector.detect_holes(
                    temp_file_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_confidence=min_confidence
                )

                # Save enhanced detections for pipeline
                import json
                enhanced_detections_path = tempfile.mktemp(suffix='.json')
                with open(enhanced_detections_path, 'w') as f:
                    json.dump(initial_detections, f)

                # Run integrated pipeline
                detections = pipeline.run_complete_pipeline(
                    image_path=temp_file_path,
                    enhanced_detections_path=enhanced_detections_path,
                    local_threshold=local_threshold,
                    openai_threshold=openai_threshold,
                    max_openai_calls=15
                )

                # Clean up temp file
                os.unlink(enhanced_detections_path)

            elif simplified_zero_shot:
                # Use simplified zero-shot pipeline (stable version)
                logger.info("Running simplified zero-shot pipeline (stable)")
                logger.info("Components: WinCLIP + Simple Masking + Heuristic Grounding")

                # Import simplified pipeline
                from simplified_zero_shot_pipeline import SimplifiedZeroShotPipeline

                # Initialize and run pipeline
                pipeline = SimplifiedZeroShotPipeline()
                image = cv2.imread(temp_file_path)

                # Use optimized thresholds
                winclip_threshold = max(0.55, local_threshold - 0.1)
                grounding_threshold = 0.45

                logger.info(f"Using simplified thresholds: WinCLIP={winclip_threshold}, Grounding={grounding_threshold}")

                # Run simplified pipeline
                pipeline_detections = pipeline.run_simplified_pipeline(
                    image,
                    winclip_threshold=winclip_threshold,
                    grounding_threshold=grounding_threshold
                )

                # Convert to standard format
                detections = []
                for det in pipeline_detections:
                    detection = {
                        "bbox": det["bbox"],
                        "confidence": float(det.get("final_confidence", 0.8)),
                        "area_pixels": float(det["bbox"]["w"] * det["bbox"]["h"]),
                        "verification_score": float(det.get("final_confidence", 0.8)),
                        "winclip_score": float(det.get("winclip_score", 0.5)),
                        "grounding_score": float(det.get("grounding_score", 0.5)),
                        "confidence_reason": det.get("confidence_reason", "unknown"),
                        "pipeline_type": "simplified_zero_shot"
                    }
                    detections.append(detection)

                logger.info(f"Simplified Zero-Shot: {len(pipeline_detections)} confirmed defects")

            elif zero_shot_pipeline:
                # Use complete zero-shot pipeline: WinCLIP + SAM2 + Florence-2 + PatchCore
                logger.info("Running complete zero-shot fabric defect detection pipeline")
                logger.info("Components: WinCLIP + SAM2 + Florence-2 + PatchCore")

                # Import zero-shot pipeline
                from zero_shot_fabric_pipeline import ZeroShotFabricPipeline

                # Initialize and run pipeline
                pipeline = ZeroShotFabricPipeline()
                image = cv2.imread(temp_file_path)

                # Use optimized thresholds for zero-shot pipeline
                winclip_threshold = max(0.65, local_threshold - 0.05)  # Slightly lower for more detection
                grounding_threshold = 0.4  # Reasonable threshold for grounding confirmation

                logger.info(f"Using zero-shot thresholds: WinCLIP={winclip_threshold}, Grounding={grounding_threshold}")

                # Run complete pipeline
                pipeline_detections = pipeline.run_zero_shot_pipeline(
                    image,
                    winclip_threshold=winclip_threshold,
                    grounding_threshold=grounding_threshold
                )

                # Convert to standard detection format
                detections = []
                for det in pipeline_detections:
                    detection = {
                        "bbox": det["bbox"],
                        "confidence": float(det.get("final_confidence", 0.8)),
                        "area_pixels": float(det["bbox"]["w"] * det["bbox"]["h"]),
                        "verification_score": float(det.get("final_confidence", 0.8)),
                        "grounding_score": float(det.get("grounding_score", 0.5)),
                        "confidence_reason": det.get("confidence_reason", "unknown"),
                        "grounding_type": det.get("grounding_type", "none"),
                        "pipeline_type": "zero_shot_complete"
                    }
                    detections.append(detection)

                logger.info(f"Zero-Shot Pipeline: Complete pipeline -> {len(detections)} confirmed defects")

                # Save debug visualization if requested
                try:
                    heatmap = pipeline.generate_winclip_heatmap(image)
                    debug_path = f"/tmp/zero_shot_debug_{int(time.time())}.png"
                    pipeline.save_debug_visualization(image, heatmap, pipeline_detections, debug_path)
                    logger.info(f"Debug visualization saved: {debug_path}")
                except Exception as e:
                    logger.warning(f"Debug visualization failed: {e}")

            elif winclip:
                # Use WinCLIP zero-shot anomaly detection for maximum accuracy
                logger.info("Running WinCLIP zero-shot anomaly detection (arXiv:2303.14814)")

                # Import WinCLIP detector
                from winclip_fabric_detector import WinCLIPFabricDetector
                from detect_holes_segmented import SegmentedHoleDetector

                # Run initial detection
                initial_detector = SegmentedHoleDetector()
                initial_detections = initial_detector.detect_holes(
                    temp_file_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_confidence=min_confidence
                )

                # Apply WinCLIP anomaly detection
                winclip_detector = WinCLIPFabricDetector()
                image = cv2.imread(temp_file_path)

                # Use optimized threshold for WinCLIP
                winclip_threshold = max(0.70, local_threshold)
                logger.info(f"Using WinCLIP threshold: {winclip_threshold}")

                detections = winclip_detector.filter_detections_winclip(
                    image,
                    initial_detections,
                    threshold=winclip_threshold
                )

                logger.info(f"WinCLIP: {len(initial_detections)} -> {len(detections)} detections")

            elif fabric_optimized:
                # Use fabric-optimized models for maximum defect detection
                logger.info("Running fabric-optimized AI for maximum hole detection")

                # Import fabric-optimized models
                from fabric_optimized_ai_filter import FabricOptimizedAIFilter
                from detect_holes_segmented import SegmentedHoleDetector

                # Run initial detection
                initial_detector = SegmentedHoleDetector()
                initial_detections = initial_detector.detect_holes(
                    temp_file_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_confidence=min_confidence
                )

                # Apply fabric-optimized AI filtering
                fabric_filter = FabricOptimizedAIFilter()
                image = cv2.imread(temp_file_path)

                # Use optimized threshold for fabric defect detection
                fabric_threshold = max(0.65, local_threshold)  # Optimized for fabric defects
                logger.info(f"Using fabric-optimized threshold: {fabric_threshold}")

                detections = fabric_filter.filter_detections_fabric_optimized(
                    image,
                    initial_detections,
                    threshold=fabric_threshold
                )

                logger.info(f"Fabric-Optimized AI: {len(initial_detections)} -> {len(detections)} detections")

            elif advanced_ai:
                # Use advanced RTX 5090 optimized AI models
                logger.info("Running advanced RTX 5090 optimized AI detection")

                # Import advanced models
                from advanced_local_ai_filter import AdvancedLocalAIFilter
                from detect_holes_segmented import SegmentedHoleDetector

                # Run initial detection
                initial_detector = SegmentedHoleDetector()
                initial_detections = initial_detector.detect_holes(
                    temp_file_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_confidence=min_confidence
                )

                # Apply advanced AI filtering with strict thresholds
                advanced_filter = AdvancedLocalAIFilter()
                image = cv2.imread(temp_file_path)

                # Use balanced threshold for advanced AI - the models are already sophisticated
                # Don't add too much to the threshold since advanced models are better at discrimination
                advanced_threshold = min(0.7, local_threshold + 0.1)  # More conservative increase
                logger.info(f"Using advanced AI threshold: {advanced_threshold}")

                detections = advanced_filter.filter_detections_ensemble(
                    image,
                    initial_detections,
                    threshold=advanced_threshold
                )

                # Apply additional size-based filtering for real holes
                size_filtered = []
                for det in detections:
                    area = det['bbox']['w'] * det['bbox']['h']
                    # Real holes are typically 200-5000 pixels
                    if 200 <= area <= 5000:
                        size_filtered.append(det)

                detections = size_filtered
                logger.info(f"Advanced AI: {len(initial_detections)} -> {len(detections)} detections")

            else:
                # Use local AI verification only
                logger.info("Running local AI verification only")
                detections = detector.detect_and_verify(
                    temp_file_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_confidence=min_confidence,
                    min_verification_score=local_threshold
                )

            # Format response
            holes = []
            for detection in detections:
                hole = {
                    "bbox": detection["bbox"],
                    "confidence": float(detection["confidence"]),
                    "area_pixels": float(detection["area_pixels"])
                }

                # Add verification score if available
                if "verification_score" in detection:
                    hole["verification_score"] = float(detection["verification_score"])
                elif "local_ai_probability" in detection:
                    hole["verification_score"] = float(detection["local_ai_probability"])

                # Add OpenAI verification if available
                if "openai_verification" in detection:
                    hole["openai_verification"] = detection["openai_verification"]

                holes.append(hole)

            processing_time = time.time() - start_time

            response = DetectionResponse(
                success=True,
                num_holes_detected=len(holes),
                holes=holes,
                message=f"Successfully detected {len(holes)} hole(s) in {processing_time:.2f}s",
                processing_time_seconds=processing_time
            )

            logger.info(f"Detection complete: {len(holes)} holes found in {processing_time:.2f}s")
            return response

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        processing_time = time.time() - start_time
        return DetectionResponse(
            success=False,
            num_holes_detected=0,
            holes=[],
            message=f"Error processing image: {str(e)}",
            processing_time_seconds=processing_time
        )

@app.post("/detect-holes-simple")
async def detect_holes_simple(image: UploadFile = File(...)):
    """
    Simplified hole detection endpoint - local AI only, no parameters

    Returns basic JSON response with bounding boxes
    """
    try:
        # Initialize detector with defaults
        initialize_detector(use_openai=False)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Run detection with default parameters
            detections = detector.detect_and_verify(
                temp_file_path,
                tile_size=512,
                overlap=128,
                min_confidence=0.7,
                min_verification_score=0.45
            )

            # Simple response format
            holes = []
            for detection in detections:
                holes.append({
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"]
                })

            return {
                "success": True,
                "holes_found": len(holes),
                "holes": holes
            }

        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass

    except Exception as e:
        logger.error(f"Error in simple detection: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "holes_found": 0,
            "holes": []
        }

if __name__ == "__main__":
    # Configuration for development/production
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print(f"🚀 Starting Garment Hole Detection API on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )