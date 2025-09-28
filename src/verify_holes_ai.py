import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from typing import List, Dict, Tuple
from detect_holes_segmented import SegmentedHoleDetector
import json


class AIHoleVerifier:
    """
    AI-powered verification to filter false positives.
    Uses pre-trained vision model to classify patches as real holes vs artifacts.
    """

    def __init__(self, model_name: str = 'microsoft/resnet-50'):
        print(f"Loading verification model: {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded successfully")

    def extract_patch_with_context(self, image: np.ndarray, bbox: Dict,
                                   context_factor: float = 1.5) -> np.ndarray:
        """
        Extract patch with surrounding context for better classification.

        Args:
            image: Original image
            bbox: Detection bounding box
            context_factor: How much context to include (1.5 = 50% more)

        Returns:
            Patch image with context
        """
        h, w = image.shape[:2]
        x, y = bbox['x'], bbox['y']
        bw, bh = bbox['w'], bbox['h']

        cx = x + bw // 2
        cy = y + bh // 2

        new_w = int(bw * context_factor)
        new_h = int(bh * context_factor)

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, cx + new_w // 2)
        y2 = min(h, cy + new_h // 2)

        patch = image[y1:y2, x1:x2].copy()

        if patch.size == 0:
            return image[y:y+bh, x:x+bw].copy()

        return patch

    def compute_semantic_score(self, patch: np.ndarray) -> float:
        """
        Compute semantic anomaly score using pre-trained model features.

        Returns:
            Score 0-1 where higher = more likely to be a real hole
        """
        if patch.size == 0:
            return 0.0

        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=patch_rgb, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            features = outputs.logits.numpy().flatten()

        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))

        uncertainty_score = float(entropy) / 5.0
        uncertainty_score = min(1.0, max(0.0, uncertainty_score))

        feature_variance = np.var(features)
        variance_score = min(1.0, feature_variance / 10.0)

        top_prob = float(torch.max(probs))
        confidence_score = 1.0 - top_prob

        semantic_score = (uncertainty_score * 0.4 +
                         variance_score * 0.3 +
                         confidence_score * 0.3)

        return semantic_score

    def compute_visual_features(self, image: np.ndarray, bbox: Dict) -> Dict[str, float]:
        """
        Compute hand-crafted visual features for the detection.
        """
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

        patch = image[y:y+h, x:x+w]
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

        context_size = 20
        x1 = max(0, x - context_size)
        y1 = max(0, y - context_size)
        x2 = min(image.shape[1], x + w + context_size)
        y2 = min(image.shape[0], y + h + context_size)
        context = image[y1:y2, x1:x2]
        gray_context = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY) if len(context.shape) == 3 else context

        patch_mean = np.mean(gray_patch)
        context_mean = np.mean(gray_context)
        brightness_diff = abs(patch_mean - context_mean) / 255.0

        edges = cv2.Canny(gray_patch, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0.0

        laplacian = cv2.Laplacian(gray_patch, cv2.CV_64F)
        texture_var = np.var(laplacian) / 1000.0

        aspect_ratio = max(w, h) / (min(w, h) + 1e-8)

        _, binary = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest)
            solidity = contour_area / (hull_area + 1e-8)
        else:
            solidity = 0.0

        return {
            'brightness_diff': brightness_diff,
            'edge_density': edge_density,
            'texture_var': min(1.0, texture_var),
            'aspect_ratio': min(1.0, aspect_ratio / 3.0),
            'solidity': solidity
        }

    def verify_detection(self, image: np.ndarray, detection: Dict,
                        use_ai: bool = True) -> Tuple[float, Dict]:
        """
        Verify if detection is a real hole or false positive.

        Args:
            image: Original image
            detection: Detection dict with bbox, confidence, area_pixels
            use_ai: Whether to use AI model (slower but more accurate)

        Returns:
            (verification_score, debug_info)
        """
        bbox = detection['bbox']

        visual_features = self.compute_visual_features(image, bbox)

        visual_score = (
            visual_features['brightness_diff'] * 0.25 +
            visual_features['edge_density'] * 0.15 +
            visual_features['texture_var'] * 0.20 +
            (1.0 - visual_features['aspect_ratio']) * 0.20 +
            (1.0 - visual_features['solidity']) * 0.20
        )

        if use_ai:
            patch = self.extract_patch_with_context(image, bbox, context_factor=1.5)
            semantic_score = self.compute_semantic_score(patch)

            final_score = visual_score * 0.4 + semantic_score * 0.6
        else:
            final_score = visual_score
            semantic_score = 0.0

        debug_info = {
            'visual_features': visual_features,
            'visual_score': visual_score,
            'semantic_score': semantic_score,
            'final_score': final_score
        }

        return final_score, debug_info


class VerifiedHoleDetector:
    """
    Complete pipeline: Segment → Tile → Detect → Verify → Filter
    """

    def __init__(self, use_ai_verification: bool = True):
        self.detector = SegmentedHoleDetector()
        self.verifier = AIHoleVerifier() if use_ai_verification else None
        self.use_ai = use_ai_verification

    def detect_and_verify(self, image_path: str,
                         tile_size: int = 512,
                         overlap: int = 128,
                         min_confidence: float = 0.7,
                         min_verification_score: float = 0.65,
                         debug_scores: bool = False) -> List[Dict]:
        """
        Complete detection pipeline with AI verification.

        Args:
            image_path: Path to image
            tile_size: Tile size for segmented detection
            overlap: Tile overlap
            min_confidence: Minimum detection confidence
            min_verification_score: Minimum AI verification score

        Returns:
            List of verified hole detections
        """
        print("=" * 70)
        print("VERIFIED HOLE DETECTION PIPELINE")
        print("=" * 70)

        print("\n[PHASE 1] Initial Detection (Segmented Approach)")
        print("-" * 70)
        detections = self.detector.detect_holes(
            image_path,
            tile_size=tile_size,
            overlap=overlap,
            min_confidence=min_confidence
        )

        print(f"\n[PHASE 2] AI Verification ({len(detections)} candidates)")
        print("-" * 70)

        if not self.verifier:
            print("Skipping AI verification (disabled)")
            return detections

        img = cv2.imread(image_path)

        verified_detections = []
        all_scores = []

        for i, det in enumerate(detections):
            print(f"Verifying detection {i+1}/{len(detections)}...", end='\r')

            verification_score, debug_info = self.verifier.verify_detection(
                img, det, use_ai=self.use_ai
            )

            det['verification_score'] = verification_score
            det['verification_debug'] = debug_info
            all_scores.append(verification_score)

            if verification_score >= min_verification_score:
                verified_detections.append(det)

        if debug_scores and all_scores:
            print(f"\n\nVerification score statistics:")
            print(f"  Min: {min(all_scores):.3f}")
            print(f"  Max: {max(all_scores):.3f}")
            print(f"  Mean: {np.mean(all_scores):.3f}")
            print(f"  Median: {np.median(all_scores):.3f}")
            print(f"  Threshold used: {min_verification_score:.3f}")

        print(f"\n\n[PHASE 3] Results")
        print("-" * 70)
        print(f"Initial detections: {len(detections)}")
        print(f"Passed verification: {len(verified_detections)}")
        print(f"False positives filtered: {len(detections) - len(verified_detections)}")
        reduction_pct = (1 - len(verified_detections)/len(detections))*100 if len(detections) > 0 else 0.0
        print(f"Reduction: {reduction_pct:.1f}%")

        verified_detections.sort(key=lambda d: d['verification_score'], reverse=True)

        return verified_detections


def draw_verified_detections(image_path: str, detections: List[Dict],
                            output_path: str = "output_verified_holes.jpg"):
    """Draw verified detections with verification scores."""
    img = cv2.imread(image_path)

    if not detections:
        cv2.putText(img, "No verified holes detected", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imwrite(output_path, img)
        print(f"Output saved to {output_path}")
        return

    for i, det in enumerate(detections):
        bbox = det['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        conf = det['confidence']
        ver_score = det.get('verification_score', 0.0)

        color = (0, 255, 0) if ver_score > 0.8 else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        label = f"HOLE #{i+1}"
        conf_label = f"Det:{conf:.2f} Ver:{ver_score:.2f}"

        cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.putText(img, conf_label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")


def test_verified_detection():
    """Test verified hole detection with AI."""
    test_image = "test_shirt.jpg"

    print("=" * 70)
    print("TESTING VERIFIED HOLE DETECTION")
    print("=" * 70)
    print(f"\nInput: {test_image}")
    print("Strategy: Segment → Tile → Detect → AI Verify → Filter\n")

    detector = VerifiedHoleDetector(use_ai_verification=True)

    detections = detector.detect_and_verify(
        test_image,
        tile_size=512,
        overlap=128,
        min_confidence=0.7,
        min_verification_score=0.52,
        debug_scores=True
    )

    print("\n" + "=" * 70)
    print(f"FINAL VERIFIED RESULTS: {len(detections)} hole(s)")
    print("=" * 70)

    if detections:
        print("\nTop 10 Verified Detections:")
        print("-" * 70)
        for i, det in enumerate(detections[:10]):
            bbox = det['bbox']
            print(f"\n[HOLE #{i+1}]")
            print(f"  Location: ({bbox['x']}, {bbox['y']})")
            print(f"  Size: {bbox['w']}x{bbox['h']} pixels")
            print(f"  Detection confidence: {det['confidence']:.2%}")
            print(f"  Verification score: {det['verification_score']:.2%}")
            print(f"  Visual score: {det['verification_debug']['visual_score']:.3f}")
            print(f"  Semantic score: {det['verification_debug']['semantic_score']:.3f}")

        draw_verified_detections(test_image, detections)

        detections_serializable = []
        for det in detections:
            det_copy = {
                'bbox': det['bbox'],
                'confidence': float(det['confidence']),
                'area_pixels': float(det['area_pixels']),
                'verification_score': float(det['verification_score']),
                'verification_debug': {
                    'visual_score': float(det['verification_debug']['visual_score']),
                    'semantic_score': float(det['verification_debug']['semantic_score']),
                    'final_score': float(det['verification_debug']['final_score']),
                    'visual_features': {k: float(v) for k, v in det['verification_debug']['visual_features'].items()}
                }
            }
            detections_serializable.append(det_copy)

        with open("verified_detections.json", "w") as f:
            json.dump(detections_serializable, f, indent=2)
        print("\n\nFull results saved to: verified_detections.json")
    else:
        print("\n✓ No holes detected after verification.")
        draw_verified_detections(test_image, detections)

    target_x, target_y = 1071, 2555
    print(f"\n{'='*70}")
    print(f"Checking for target hole at ({target_x}, {target_y})...")
    print('-'*70)
    for i, det in enumerate(detections):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        if dist < 150:
            print(f"✓ FOUND target hole as verified detection #{i+1}!")
            print(f"  Distance: {dist:.0f}px")
            print(f"  Detection confidence: {det['confidence']:.2%}")
            print(f"  Verification score: {det['verification_score']:.2%}")

    return detections


if __name__ == "__main__":
    test_verified_detection()