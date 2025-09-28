import cv2
import numpy as np
from typing import List, Dict
from verify_holes_enhanced import VerifiedHoleDetector, EnhancedHoleFilter, draw_verified_detections
import json


class ImprovedHoleScorer:
    """
    Redesigned scoring system that properly prioritizes real hole characteristics.
    """

    def compute_hole_specific_features(self, image: np.ndarray, detection: Dict) -> Dict:
        """Compute features that distinguish real holes from decorative patterns."""
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

        # Extract patch and context
        patch = image[y:y+h, x:x+w]
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

        # Larger context for comparison
        context_size = max(w, h) * 2
        x1 = max(0, x - context_size)
        y1 = max(0, y - context_size)
        x2 = min(image.shape[1], x + w + context_size)
        y2 = min(image.shape[0], y + h + context_size)
        context = image[y1:y2, x1:x2]
        gray_context = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY) if len(context.shape) == 3 else context

        # 1. DEPTH ANALYSIS - Real holes show much darker backgrounds
        patch_min = np.min(gray_patch)
        patch_mean = np.mean(gray_patch)
        context_mean = np.mean(gray_context)

        # How much darker is the darkest part vs surroundings?
        depth_contrast = (context_mean - patch_min) / 255.0

        # 2. BACKGROUND VISIBILITY - Real holes show the surface underneath
        # Count pixels significantly darker than context
        dark_threshold = context_mean - 30  # 30 gray levels darker
        dark_pixels = np.sum(gray_patch < dark_threshold)
        background_visibility = dark_pixels / gray_patch.size if gray_patch.size > 0 else 0.0

        # 3. EDGE IRREGULARITY - Real holes have torn/irregular edges
        edges = cv2.Canny(gray_patch, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Measure how irregular the contour is
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if area > 0:
                # Higher ratio = more irregular (jagged edges)
                irregularity = (perimeter * perimeter) / (4 * np.pi * area)
            else:
                irregularity = 1.0
        else:
            irregularity = 1.0

        # 4. TEXTURE DISRUPTION - Real holes break fabric patterns
        # Use Laplacian to detect texture breaks
        laplacian = cv2.Laplacian(gray_patch, cv2.CV_64F)
        texture_disruption = np.var(laplacian) / 1000.0

        # 5. COLOR UNIFORMITY - Decorative dots have consistent color
        # Real holes show diverse colors (fabric + background)
        if len(patch.shape) == 3:
            # Calculate color variance across channels
            color_vars = [np.var(patch[:,:,i]) for i in range(3)]
            color_diversity = np.mean(color_vars) / 10000.0
        else:
            color_diversity = np.var(gray_patch) / 10000.0

        # 6. SHAPE ANALYSIS - Decorative dots are round, holes are irregular
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            if hull_area > 0:
                # Lower solidity = more irregular shape
                solidity = contour_area / hull_area
                shape_irregularity = 1.0 - solidity
            else:
                shape_irregularity = 0.5
        else:
            shape_irregularity = 0.5

        return {
            'depth_contrast': min(1.0, depth_contrast),
            'background_visibility': min(1.0, background_visibility),
            'irregularity': min(1.0, irregularity / 5.0),  # Normalize
            'texture_disruption': min(1.0, texture_disruption),
            'color_diversity': min(1.0, color_diversity),
            'shape_irregularity': shape_irregularity,
            'patch_min_intensity': patch_min / 255.0,
            'context_mean_intensity': context_mean / 255.0
        }

    def compute_hole_score(self, image: np.ndarray, detection: Dict) -> float:
        """
        Compute hole-specific score. Higher score = more likely to be a real hole.
        """
        hole_features = self.compute_hole_specific_features(image, detection)

        # Weight features that distinguish real holes from decorative patterns
        score = (
            hole_features['depth_contrast'] * 0.30 +          # Dark background visible
            hole_features['background_visibility'] * 0.25 +   # Clear background pixels
            hole_features['shape_irregularity'] * 0.20 +      # Irregular torn shape
            hole_features['texture_disruption'] * 0.15 +      # Breaks fabric pattern
            hole_features['color_diversity'] * 0.10           # Mixed colors (fabric+background)
        )

        # Bonus for appropriate size (not too tiny, not huge)
        area = detection['area_pixels']
        if 300 <= area <= 3000:  # Good hole size range
            size_bonus = 1.1
        elif 100 <= area <= 5000:  # Acceptable range
            size_bonus = 1.05
        else:
            size_bonus = 0.9

        final_score = score * size_bonus

        # Store detailed analysis
        detection['hole_analysis'] = hole_features
        detection['hole_score'] = final_score

        return final_score


def test_improved_scoring():
    """Test the improved hole-focused scoring system."""
    test_image = "test_shirt.jpg"

    print("=" * 70)
    print("IMPROVED HOLE DETECTION - HOLE-FOCUSED SCORING")
    print("=" * 70)

    # Get detections from previous pipeline
    with open('enhanced_detections.json', 'r') as f:
        detections = json.load(f)

    print(f"Rescoring {len(detections)} detections with hole-focused algorithm...")

    img = cv2.imread(test_image)
    scorer = ImprovedHoleScorer()

    # Rescore all detections
    for det in detections:
        new_score = scorer.compute_hole_score(img, det)
        det['original_score'] = det['final_confidence_score']
        det['final_confidence_score'] = new_score

    # Re-sort by new scores
    detections.sort(key=lambda d: d['final_confidence_score'], reverse=True)

    print("\n" + "=" * 70)
    print("NEW RANKINGS WITH HOLE-FOCUSED SCORING")
    print("=" * 70)

    # Find where actual hole ranks now
    target_x, target_y = 1660, 2482
    actual_hole_rank = None

    for i, det in enumerate(detections):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        if dist < 50:
            actual_hole_rank = i + 1
            print(f"\n🎯 ACTUAL HOLE NOW RANKED: #{actual_hole_rank}")
            print(f"   Location: ({bbox['x']}, {bbox['y']})")
            print(f"   Original score: {det['original_score']:.3f} (rank #{60})")
            print(f"   NEW score: {det['final_confidence_score']:.3f} (rank #{actual_hole_rank})")

            if 'hole_analysis' in det:
                ha = det['hole_analysis']
                print(f"   Depth contrast: {ha['depth_contrast']:.3f}")
                print(f"   Background visibility: {ha['background_visibility']:.3f}")
                print(f"   Shape irregularity: {ha['shape_irregularity']:.3f}")
            break

    print(f"\n\nTOP 10 DETECTIONS (New Scoring):")
    print("-" * 70)
    for i in range(min(10, len(detections))):
        det = detections[i]
        bbox = det['bbox']
        new_score = det['final_confidence_score']
        old_score = det.get('original_score', 0)

        print(f"\n#{i+1}: ({bbox['x']}, {bbox['y']}) {bbox['w']}x{bbox['h']}")
        print(f"   NEW score: {new_score:.3f} (was: {old_score:.3f})")

        if 'hole_analysis' in det:
            ha = det['hole_analysis']
            print(f"   Depth: {ha['depth_contrast']:.3f} | Background: {ha['background_visibility']:.3f} | Irregular: {ha['shape_irregularity']:.3f}")

    # Filter top candidates
    top_candidates = [d for d in detections if d['final_confidence_score'] > 0.4]

    print(f"\n\nSUMMARY:")
    print(f"  Actual hole rank: #{actual_hole_rank if actual_hole_rank else 'Not found'}")
    print(f"  High-confidence candidates (>0.4): {len(top_candidates)}")

    # Save improved results
    with open("improved_hole_detections.json", "w") as f:
        json.dump(detections[:20], f, indent=2)  # Save top 20

    # Visualize top detections
    draw_verified_detections(test_image, detections[:20], "output_improved_holes.jpg")
    print(f"  Visualization saved: output_improved_holes.jpg")
    print(f"  Results saved: improved_hole_detections.json")

    return detections


if __name__ == "__main__":
    test_improved_scoring()