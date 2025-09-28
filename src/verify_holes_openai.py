import cv2
import numpy as np
import json
import base64
import time
from io import BytesIO
from PIL import Image
from typing import List, Dict
from verify_holes_enhanced import VerifiedHoleDetector, EnhancedHoleFilter, draw_verified_detections
import openai


class OpenAIHoleVerifier:
    """
    OpenAI Vision API verifier for final hole validation.
    Uses GPT-4V to distinguish real holes from false positives.
    """

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.verification_prompt = """Analyze this fabric image patch carefully.

You are looking at a small region from a striped children's shirt with decorative pink dots. Your task is to determine if this patch contains a REAL HOLE or is a FALSE POSITIVE.

REAL HOLE characteristics:
- Actual gap/tear in the fabric
- Shows darker background/surface underneath
- Has irregular or torn edges
- Missing fabric material

FALSE POSITIVE characteristics:
- Decorative pink dots (part of the pattern)
- Fabric folds or creases
- Shadows
- Normal striped pattern
- Seams or stitching

Respond ONLY with valid JSON in this exact format:
{"is_hole": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}"""

    def encode_image_to_base64(self, image_patch: np.ndarray) -> str:
        """Convert OpenCV image to base64 for API."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64

    def verify_single_region(self, image_patch: np.ndarray, detection_info: Dict = None) -> Dict:
        """
        Verify a single image patch using OpenAI Vision API.

        Args:
            image_patch: Image region to verify
            detection_info: Optional info about the detection

        Returns:
            Dict with verification results
        """
        try:
            image_base64 = self.encode_image_to_base64(image_patch)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.verification_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                max_tokens=150,
                temperature=0.1
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Validate required fields
            if not all(key in result for key in ["is_hole", "confidence", "reason"]):
                raise ValueError("Missing required fields in response")

            return {
                "is_hole": bool(result["is_hole"]),
                "confidence": float(result["confidence"]),
                "reason": str(result["reason"]),
                "api_success": True
            }

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {content}")
            return {
                "is_hole": False,
                "confidence": 0.0,
                "reason": f"API response parsing error: {str(e)}",
                "api_success": False
            }
        except Exception as e:
            print(f"API error: {e}")
            return {
                "is_hole": False,
                "confidence": 0.0,
                "reason": f"API error: {str(e)}",
                "api_success": False
            }

    def verify_detections_batch(self, image: np.ndarray, detections: List[Dict],
                               max_detections: int = 20,
                               min_openai_confidence: float = 0.7) -> List[Dict]:
        """
        Verify multiple detections using OpenAI Vision API.

        Args:
            image: Original image
            detections: List of detections to verify
            max_detections: Maximum number to send to API (cost control)
            min_openai_confidence: Minimum OpenAI confidence to keep detection

        Returns:
            List of verified detections
        """
        print(f"\n[OPENAI VISION VERIFICATION]")
        print("-" * 70)
        print(f"Verifying top {min(max_detections, len(detections))} detections with OpenAI...")

        # Take top detections by confidence
        sorted_detections = sorted(detections, key=lambda d: d['final_confidence_score'], reverse=True)
        candidates = sorted_detections[:max_detections]

        verified_detections = []
        api_calls = 0

        for i, detection in enumerate(candidates):
            print(f"  Verifying detection {i+1}/{len(candidates)}...", end='')

            # Extract region with some context
            bbox = detection['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

            # Use FIXED minimum context size for OpenAI (regardless of detection size)
            min_context_size = 80  # Always extract at least 160x160 region

            cx = x + w // 2
            cy = y + h // 2

            x1 = max(0, cx - min_context_size)
            y1 = max(0, cy - min_context_size)
            x2 = min(image.shape[1], cx + min_context_size)
            y2 = min(image.shape[0], cy + min_context_size)

            patch = image[y1:y2, x1:x2]

            if patch.size == 0:
                print(" [SKIP - invalid region]")
                continue

            # Call OpenAI API
            api_result = self.verify_single_region(patch, detection)
            api_calls += 1

            # Add OpenAI results to detection
            detection['openai_verification'] = api_result

            # Check if verified as real hole
            if api_result['is_hole'] and api_result['confidence'] >= min_openai_confidence:
                verified_detections.append(detection)
                print(f" ✓ REAL HOLE (conf: {api_result['confidence']:.2f})")
            else:
                reason = api_result['reason'][:50] + "..." if len(api_result['reason']) > 50 else api_result['reason']
                print(f" ✗ FALSE POSITIVE ({reason})")

            # Rate limiting
            time.sleep(0.5)  # Be nice to the API

        print(f"\nOpenAI API Summary:")
        print(f"  Total API calls: {api_calls}")
        print(f"  Estimated cost: ~${api_calls * 0.01:.2f}")
        print(f"  Verified as real holes: {len(verified_detections)}")
        print(f"  False positives filtered: {len(candidates) - len(verified_detections)}")

        return verified_detections


class UltimateHoleDetector:
    """
    Complete hole detection pipeline with OpenAI verification.
    """

    def __init__(self, openai_api_key: str):
        self.base_detector = VerifiedHoleDetector(use_ai_verification=True)
        self.enhancer = EnhancedHoleFilter()
        self.openai_verifier = OpenAIHoleVerifier(openai_api_key)

    def detect_with_openai_verification(self, image_path: str,
                                      max_openai_verifications: int = 15,
                                      min_openai_confidence: float = 0.7) -> List[Dict]:
        """
        Ultimate hole detection pipeline with OpenAI verification.

        Args:
            image_path: Path to image
            max_openai_verifications: Max detections to send to OpenAI (cost control)
            min_openai_confidence: Minimum OpenAI confidence to keep

        Returns:
            List of OpenAI-verified hole detections
        """
        print("=" * 70)
        print("ULTIMATE HOLE DETECTION WITH OPENAI VERIFICATION")
        print("=" * 70)
        print(f"Input: {image_path}")
        print("Pipeline: Segment → Tile → Detect → AI Verify → Enhanced Filter → OpenAI Vision")

        # Phase 1: Standard detection pipeline
        from detect_holes_segmented import SegmentedHoleDetector as BaseDetector

        base_detector = BaseDetector()
        img = cv2.imread(image_path)

        print("\n[PHASE 1] Segmented Detection")
        print("-" * 70)
        mask, bbox = base_detector.segment_garment(img)
        tiles = base_detector.create_tiles(img, mask, bbox, tile_size=256, overlap=64)

        all_detections = []
        for i, tile in enumerate(tiles):
            print(f"Processing tile {i+1}/{len(tiles)}...", end='\r')
            tile_detections = base_detector.detect_holes_in_tile(
                tile['image'], tile['mask'], patch_size=32, stride=16, contamination=0.08
            )
            for det in tile_detections:
                det['bbox']['x'] += tile['x_offset']
                det['bbox']['y'] += tile['y_offset']
                all_detections.append(det)

        initial_detections = base_detector.merge_overlapping_detections(all_detections, iou_threshold=0.3)
        initial_detections = [d for d in initial_detections if d['confidence'] >= 0.7]
        print(f"\nInitial detections: {len(initial_detections)}")

        # Phase 2: AI Verification
        print("\n[PHASE 2] AI Verification (ResNet-50)")
        print("-" * 70)
        if self.base_detector.verifier:
            verified_detections = []
            for i, det in enumerate(initial_detections):
                verification_score, debug_info = self.base_detector.verifier.verify_detection(
                    img, det, use_ai=True
                )
                det['verification_score'] = verification_score
                det['verification_debug'] = debug_info
                if verification_score >= 0.45:
                    verified_detections.append(det)
            print(f"After AI verification: {len(verified_detections)}")
        else:
            verified_detections = initial_detections

        # Phase 3: Enhanced Filtering
        print("\n[PHASE 3] Enhanced Filtering")
        print("-" * 70)
        enhanced_detections = self.enhancer.apply_enhanced_filters(
            img, verified_detections, min_final_score=0.30, strict_mode=False
        )

        # Phase 4: OpenAI Vision Verification
        final_detections = self.openai_verifier.verify_detections_batch(
            img, enhanced_detections,
            max_detections=max_openai_verifications,
            min_openai_confidence=min_openai_confidence
        )

        print(f"\n{'='*70}")
        print(f"FINAL RESULTS: {len(final_detections)} OpenAI-verified hole(s)")
        print(f"{'='*70}")

        # Pipeline summary
        print(f"\nPipeline Summary:")
        print(f"  Initial detections: {len(initial_detections)}")
        print(f"  After AI verification: {len(verified_detections)}")
        print(f"  After enhanced filtering: {len(enhanced_detections)}")
        print(f"  After OpenAI verification: {len(final_detections)}")
        reduction_pct = (1 - len(final_detections)/len(initial_detections))*100 if len(initial_detections) > 0 else 0.0
        print(f"  Total reduction: {reduction_pct:.1f}%")

        return final_detections


def test_openai_verification(api_key: str):
    """Test OpenAI verification on the updated image."""
    test_image = "test_shirt.jpg"

    detector = UltimateHoleDetector(api_key)

    final_detections = detector.detect_with_openai_verification(
        test_image,
        max_openai_verifications=15,  # Limit API calls for cost control
        min_openai_confidence=0.7
    )

    if final_detections:
        print(f"\nOpenAI-Verified Detections:")
        print("-" * 70)
        for i, det in enumerate(final_detections):
            bbox = det['bbox']
            openai_result = det['openai_verification']

            print(f"\n[VERIFIED HOLE #{i+1}]")
            print(f"  Location: ({bbox['x']}, {bbox['y']})")
            print(f"  Size: {bbox['w']}x{bbox['h']} = {det['area_pixels']:.0f}px²")
            print(f"  Detection confidence: {det['confidence']:.2%}")
            print(f"  Final score: {det['final_confidence_score']:.3f}")
            print(f"  OpenAI confidence: {openai_result['confidence']:.2%}")
            print(f"  OpenAI reason: {openai_result['reason']}")

        # Save visualization
        draw_verified_detections(test_image, final_detections, "output_openai_verified.jpg")
        print(f"\n\nVisualization saved to: output_openai_verified.jpg")

        # Save results
        serializable_detections = []
        for det in final_detections:
            det_copy = {
                'bbox': det['bbox'],
                'confidence': float(det['confidence']),
                'area_pixels': float(det['area_pixels']),
                'final_confidence_score': float(det['final_confidence_score']),
                'openai_verification': det['openai_verification']
            }
            serializable_detections.append(det_copy)

        with open("openai_verified_detections.json", "w") as f:
            json.dump(serializable_detections, f, indent=2)
        print("Results saved to: openai_verified_detections.json")

    else:
        print("\n✓ No holes verified by OpenAI Vision API.")
        draw_verified_detections(test_image, final_detections, "output_openai_verified.jpg")

    # Check for target hole
    target_x, target_y = 1660, 2482
    print(f"\n{'='*70}")
    print(f"Checking for target hole at ({target_x}, {target_y})...")
    print('-'*70)
    found = False
    for i, det in enumerate(final_detections):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        if dist < 150:
            print(f"✓ FOUND target hole as verified detection #{i+1}!")
            print(f"  Distance: {dist:.0f}px")
            print(f"  OpenAI confidence: {det['openai_verification']['confidence']:.2%}")
            print(f"  OpenAI reason: {det['openai_verification']['reason']}")
            found = True

    if not found:
        print("✗ Target hole was not verified by OpenAI (might have been filtered out)")

    return final_detections


if __name__ == "__main__":
    print("OpenAI Vision Hole Verifier Ready!")
    print("Please provide your OpenAI API key to run the test.")