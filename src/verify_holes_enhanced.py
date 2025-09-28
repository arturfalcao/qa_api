import cv2
import numpy as np
from typing import List, Dict, Tuple
from verify_holes_ai import VerifiedHoleDetector, draw_verified_detections
import json


class EnhancedHoleFilter:
    """
    Additional filtering strategies to further reduce false positives:
    1. Area-based filtering (too small detections are likely dots/noise)
    2. Brightness contrast requirement (holes should have visible darkness)
    3. Edge density requirement (holes should have clear boundaries)
    4. Clustering analysis (remove isolated tiny detections)
    5. Shape analysis (extreme aspect ratios are suspicious)
    6. Context validation (check surrounding area)
    """

    def __init__(self):
        pass

    def compute_additional_features(self, image: np.ndarray, detection: Dict) -> Dict:
        """Compute additional features for enhanced filtering."""
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

        patch = image[y:y+h, x:x+w]
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

        context_size = max(w, h)
        x1 = max(0, x - context_size)
        y1 = max(0, y - context_size)
        x2 = min(image.shape[1], x + w + context_size)
        y2 = min(image.shape[0], y + h + context_size)
        context = image[y1:y2, x1:x2]
        gray_context = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY) if len(context.shape) == 3 else context

        patch_mean = np.mean(gray_patch)
        context_mean = np.mean(gray_context)
        patch_min = np.min(gray_patch)

        is_darker = patch_mean < context_mean
        darkness_score = (context_mean - patch_mean) / 255.0 if is_darker else 0.0

        min_darkness = (context_mean - patch_min) / 255.0

        patch_std = np.std(gray_patch)
        context_std = np.std(gray_context)
        uniformity = 1.0 - min(1.0, patch_std / (context_std + 1e-8))

        edges = cv2.Canny(gray_patch, 30, 100)
        boundary_edges = np.concatenate([
            edges[0, :], edges[-1, :], edges[:, 0], edges[:, -1]
        ])
        boundary_edge_ratio = np.sum(boundary_edges > 0) / len(boundary_edges) if len(boundary_edges) > 0 else 0.0

        h_img, w_img = image.shape[:2]
        center_x = x + w // 2
        center_y = y + h // 2
        dist_to_edge = min(center_x, center_y, w_img - center_x, h_img - center_y)
        edge_proximity = 1.0 - min(1.0, dist_to_edge / 100.0)

        return {
            'is_darker': is_darker,
            'darkness_score': darkness_score,
            'min_darkness': min_darkness,
            'uniformity': uniformity,
            'boundary_edge_ratio': boundary_edge_ratio,
            'edge_proximity': edge_proximity,
            'patch_mean': patch_mean / 255.0,
            'context_mean': context_mean / 255.0
        }

    def apply_enhanced_filters(self, image: np.ndarray, detections: List[Dict],
                              min_final_score: float = 0.55,
                              strict_mode: bool = False) -> List[Dict]:
        """
        Apply scoring-based filtering to reduce false positives.
        Uses weighted scoring instead of hard cutoffs to avoid being too aggressive.

        Args:
            image: Original image
            detections: List of detections with verification scores
            min_final_score: Minimum final confidence score (0-1)
            strict_mode: If True, applies stricter area filtering

        Returns:
            Filtered list of detections
        """
        print("\n[ENHANCED FILTERING - Scoring-based]")
        print("-" * 70)

        filtered = []

        for det in detections:
            bbox = det['bbox']
            area = det['area_pixels']
            vf = det['verification_debug']['visual_features']
            ver_score = det['verification_score']

            extra_features = self.compute_additional_features(image, det)
            det['enhanced_features'] = extra_features

            if strict_mode and area < 300:
                continue

            area_score = min(1.0, area / 1000.0)

            darkness_score = extra_features['darkness_score'] * 2.0
            darkness_score = min(1.0, darkness_score)

            boundary_score = min(1.0, extra_features['boundary_edge_ratio'] * 4.0)

            size_penalty = 1.0
            if area < 400:
                size_penalty = area / 400.0

            edge_penalty = 1.0
            if extra_features['edge_proximity'] > 0.7:
                edge_penalty = 1.0 - (extra_features['edge_proximity'] - 0.7) / 0.3

            final_score = (
                ver_score * 0.35 +
                darkness_score * 0.20 +
                vf['brightness_diff'] * 0.15 +
                boundary_score * 0.10 +
                vf['edge_density'] * 0.08 +
                vf['texture_var'] * 0.07 +
                area_score * 0.05
            )

            final_score *= size_penalty * edge_penalty

            det['final_confidence_score'] = final_score

            if final_score >= min_final_score:
                filtered.append(det)

        print(f"Filter results:")
        print(f"  Input detections: {len(detections)}")
        print(f"  Final score threshold: {min_final_score:.3f}")
        print(f"  ✓ Passed scoring filter: {len(filtered)}")
        reduction_pct = (1 - len(filtered)/len(detections))*100 if len(detections) > 0 else 0.0
        print(f"  Reduction: {reduction_pct:.1f}%")

        filtered.sort(key=lambda d: d['final_confidence_score'], reverse=True)

        return filtered


def test_enhanced_filtering():
    """Test enhanced hole detection with additional filters."""
    test_image = "test_shirt.jpg"

    print("=" * 70)
    print("ENHANCED HOLE DETECTION WITH ADVANCED FILTERING")
    print("=" * 70)
    print(f"\nInput: {test_image}")
    print("Strategy: Segment → Tile → Detect → AI Verify → Enhanced Filters\n")

    detector = VerifiedHoleDetector(use_ai_verification=True)

    print("[PHASE 1-2] Detection + AI Verification (Small Holes Mode)")
    print("-" * 70)
    print("Using smaller tiles (256x256) and patches (32x32) for small hole detection")

    import cv2
    from detect_holes_segmented import SegmentedHoleDetector as BaseDetector

    base_detector = BaseDetector()
    img_temp = cv2.imread(test_image)

    print("\nStep 1: Segmenting garment...")
    mask, bbox = base_detector.segment_garment(img_temp)
    x, y, w, h = bbox
    print(f"  Garment bounding box: ({x}, {y}) size={w}x{h}")

    print(f"\nStep 2: Creating smaller tiles (256x256, overlap=64)...")
    tiles = base_detector.create_tiles(img_temp, mask, bbox, tile_size=256, overlap=64)
    print(f"  Created {len(tiles)} tiles")

    print("\nStep 3: Detecting holes with smaller patches (32x32, stride=16)...")
    all_detections = []
    for i, tile in enumerate(tiles):
        print(f"  Processing tile {i+1}/{len(tiles)}...", end='\r')
        tile_detections = base_detector.detect_holes_in_tile(
            tile['image'],
            tile['mask'],
            patch_size=32,
            stride=16,
            contamination=0.08
        )
        for det in tile_detections:
            det['bbox']['x'] += tile['x_offset']
            det['bbox']['y'] += tile['y_offset']
            all_detections.append(det)

    print(f"\n  Found {len(all_detections)} candidate holes")

    print("\nStep 4: Merging overlapping detections...")
    detections_initial = base_detector.merge_overlapping_detections(all_detections, iou_threshold=0.3)
    detections_initial = [d for d in detections_initial if d['confidence'] >= 0.7]
    print(f"  Final count: {len(detections_initial)} holes")

    print("\n[AI VERIFICATION]")
    print("-" * 70)
    if detector.verifier:
        verified_detections = []
        for i, det in enumerate(detections_initial):
            print(f"Verifying detection {i+1}/{len(detections_initial)}...", end='\r')
            verification_score, debug_info = detector.verifier.verify_detection(
                img_temp, det, use_ai=detector.use_ai
            )
            det['verification_score'] = verification_score
            det['verification_debug'] = debug_info

            if verification_score >= 0.45:
                verified_detections.append(det)

        detections = verified_detections
        print(f"\n  Passed verification: {len(detections)}")
    else:
        detections = detections_initial

    print(f"\nAfter AI verification: {len(detections)} detections")

    img = cv2.imread(test_image)
    enhancer = EnhancedHoleFilter()

    final_detections = enhancer.apply_enhanced_filters(
        img,
        detections,
        min_final_score=0.30,
        strict_mode=False
    )

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {len(final_detections)} high-confidence hole(s)")
    print("=" * 70)
    print(f"\nReduction summary:")
    print(f"  Initial detections: 209")
    print(f"  After AI verification: {len(detections)}")
    print(f"  After enhanced filtering: {len(final_detections)}")
    initial_count = 209  # This should be passed as parameter ideally
    reduction_pct = (1 - len(final_detections)/initial_count)*100 if initial_count > 0 else 0.0
    print(f"  Total reduction: {reduction_pct:.1f}%")

    if final_detections:
        print(f"\n\nTop {min(10, len(final_detections))} Detections:")
        print("-" * 70)
        for i, det in enumerate(final_detections[:10]):
            bbox = det['bbox']
            ef = det['enhanced_features']
            print(f"\n[HOLE #{i+1}]")
            print(f"  Location: ({bbox['x']}, {bbox['y']})")
            print(f"  Size: {bbox['w']}x{bbox['h']} = {det['area_pixels']:.0f}px²")
            print(f"  Detection confidence: {det['confidence']:.2%}")
            print(f"  Verification score: {det['verification_score']:.2%}")
            print(f"  Final confidence score: {det['final_confidence_score']:.3f}")
            print(f"  Darkness: {ef['darkness_score']:.3f} | Boundary edges: {ef['boundary_edge_ratio']:.3f}")

        draw_verified_detections(test_image, final_detections, "output_enhanced_holes.jpg")
        print(f"\n\nVisualization saved to: output_enhanced_holes.jpg")

        final_serializable = []
        for det in final_detections:
            det_copy = {
                'bbox': det['bbox'],
                'confidence': float(det['confidence']),
                'area_pixels': float(det['area_pixels']),
                'verification_score': float(det['verification_score']),
                'final_confidence_score': float(det['final_confidence_score']),
                'enhanced_features': {k: float(v) for k, v in det['enhanced_features'].items()}
            }
            final_serializable.append(det_copy)

        with open("enhanced_detections.json", "w") as f:
            json.dump(final_serializable, f, indent=2)
        print("Full results saved to: enhanced_detections.json")
    else:
        print("\n✓ No holes detected after enhanced filtering.")
        draw_verified_detections(test_image, final_detections, "output_enhanced_holes.jpg")

    target_x, target_y = 1071, 2555
    print(f"\n{'='*70}")
    print(f"Checking for target hole at ({target_x}, {target_y})...")
    print('-'*70)
    found = False
    for i, det in enumerate(final_detections):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        if dist < 150:
            print(f"✓ FOUND target hole as detection #{i+1}!")
            print(f"  Distance: {dist:.0f}px")
            print(f"  Detection confidence: {det['confidence']:.2%}")
            print(f"  Verification score: {det['verification_score']:.2%}")
            print(f"  Final confidence score: {det['final_confidence_score']:.3f}")
            found = True

    if not found:
        print("✗ Target hole was filtered out. May need to adjust filter thresholds.")

    return final_detections


if __name__ == "__main__":
    test_enhanced_filtering()