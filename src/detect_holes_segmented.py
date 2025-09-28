import cv2
import numpy as np
import json
from typing import List, Dict, Tuple
from sklearn.neighbors import LocalOutlierFactor


class SegmentedHoleDetector:
    """
    Multi-stage hole detection:
    1. Segment garment from background
    2. Crop garment into small tiles
    3. Detect holes in each tile independently
    4. Merge results back to original coordinates
    """

    def segment_garment(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Segment garment from background.

        Returns:
            mask: Binary mask of garment (255 = garment, 0 = background)
            bbox: (x, y, w, h) bounding box of garment
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.ones(gray.shape, dtype=np.uint8) * 255, (0, 0, image.shape[1], image.shape[0])

        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(largest_contour)

        return mask, (x, y, w, h)

    def create_tiles(self, image: np.ndarray, mask: np.ndarray,
                     bbox: Tuple[int, int, int, int],
                     tile_size: int = 512,
                     overlap: int = 128) -> List[Dict]:
        """
        Create overlapping tiles of the garment region.

        Args:
            image: Original image
            mask: Garment mask
            bbox: Garment bounding box (x, y, w, h)
            tile_size: Size of each tile
            overlap: Overlap between tiles

        Returns:
            List of tile dictionaries with image, mask, and coordinates
        """
        x, y, w, h = bbox
        stride = tile_size - overlap

        tiles = []

        for ty in range(y, y + h, stride):
            for tx in range(x, x + w, stride):
                x1 = tx
                y1 = ty
                x2 = min(tx + tile_size, x + w)
                y2 = min(ty + tile_size, y + h)

                if x2 - x1 < tile_size // 2 or y2 - y1 < tile_size // 2:
                    continue

                tile_img = image[y1:y2, x1:x2].copy()
                tile_mask = mask[y1:y2, x1:x2].copy()

                garment_ratio = np.sum(tile_mask == 255) / tile_mask.size if tile_mask.size > 0 else 0.0
                if garment_ratio < 0.3:
                    continue

                tiles.append({
                    'image': tile_img,
                    'mask': tile_mask,
                    'x_offset': x1,
                    'y_offset': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                })

        return tiles

    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract features from a small patch."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

        features = []

        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        features.extend([mean_intensity, std_intensity])

        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_mean = np.mean(gradient_mag)
        edge_std = np.std(gradient_mag)
        features.extend([edge_mean, edge_std])

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_mean = np.mean(np.abs(laplacian))
        lap_var = np.var(laplacian)
        features.extend([lap_mean, lap_var])

        if len(patch.shape) == 3:
            for c in range(3):
                channel = patch[:, :, c]
                features.append(np.mean(channel))
                features.append(np.std(channel))

        return np.array(features, dtype=np.float32)

    def detect_holes_in_tile(self, tile_img: np.ndarray, tile_mask: np.ndarray,
                             patch_size: int = 48,
                             stride: int = 24,
                             contamination: float = 0.08) -> List[Dict]:
        """
        Detect holes within a single tile using LOF.

        Args:
            tile_img: Tile image
            tile_mask: Tile mask (garment region)
            patch_size: Size of analysis patches
            stride: Stride between patches
            contamination: Expected anomaly proportion

        Returns:
            List of detections with local coordinates
        """
        h, w = tile_img.shape[:2]

        features_list = []
        positions = []

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_mask = tile_mask[y:y+patch_size, x:x+patch_size]
                garment_ratio = np.sum(patch_mask == 255) / patch_mask.size if patch_mask.size > 0 else 0.0

                if garment_ratio < 0.5:
                    continue

                patch = tile_img[y:y+patch_size, x:x+patch_size]
                features = self.extract_patch_features(patch)
                features_list.append(features)
                positions.append((x, y, patch_size, patch_size))

        if len(features_list) < 10:
            return []

        features_array = np.array(features_list)

        lof = LocalOutlierFactor(n_neighbors=min(15, len(features_array)-1),
                                  contamination=contamination,
                                  novelty=False)
        predictions = lof.fit_predict(features_array)
        anomaly_scores = -lof.negative_outlier_factor_

        anomaly_map = np.zeros(tile_img.shape[:2], dtype=np.float32)
        count_map = np.zeros(tile_img.shape[:2], dtype=np.int32)

        for score, (x, y, pw, ph) in zip(anomaly_scores, positions):
            anomaly_map[y:y+ph, x:x+pw] += score
            count_map[y:y+ph, x:x+pw] += 1

        count_map[count_map == 0] = 1
        anomaly_map = anomaly_map / count_map

        threshold = np.percentile(anomaly_scores[predictions == -1], 60) if np.sum(predictions == -1) > 0 else np.percentile(anomaly_scores, 92)

        anomaly_binary = (anomaly_map > threshold * 0.9).astype(np.uint8) * 255
        anomaly_binary = cv2.bitwise_and(anomaly_binary, tile_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_binary = cv2.morphologyEx(anomaly_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        anomaly_binary = cv2.morphologyEx(anomaly_binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(anomaly_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 200 or area > 50000:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            mask_region = np.zeros(anomaly_map.shape, dtype=np.uint8)
            cv2.drawContours(mask_region, [contour], -1, 255, -1)

            masked_scores = anomaly_map * (mask_region / 255.0)
            mean_score = np.sum(masked_scores) / (np.sum(mask_region / 255.0) + 1e-8)

            confidence = min(1.0, mean_score / (threshold + 1e-8))

            if confidence < 0.7:
                continue

            detections.append({
                'bbox': {'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh)},
                'confidence': float(confidence),
                'area_pixels': float(area)
            })

        return detections

    def detect_holes(self, image_path: str,
                     tile_size: int = 512,
                     overlap: int = 128,
                     min_confidence: float = 0.7) -> List[Dict]:
        """
        Main detection pipeline.

        Args:
            image_path: Path to input image
            tile_size: Size of tiles to process
            overlap: Overlap between tiles
            min_confidence: Minimum confidence threshold

        Returns:
            List of detections in original image coordinates
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        print("Step 1: Segmenting garment from background...")
        mask, bbox = self.segment_garment(img)
        x, y, w, h = bbox
        print(f"  Garment bounding box: ({x}, {y}) size={w}x{h}")

        cv2.imwrite("debug_garment_mask.jpg", mask)

        masked_img = img.copy()
        masked_img[mask == 0] = [0, 0, 0]
        cv2.imwrite("debug_garment_segmented.jpg", masked_img)

        print(f"\nStep 2: Creating tiles (size={tile_size}, overlap={overlap})...")
        tiles = self.create_tiles(img, mask, bbox, tile_size, overlap)
        print(f"  Created {len(tiles)} tiles")

        print("\nStep 3: Detecting holes in each tile...")
        all_detections = []

        for i, tile in enumerate(tiles):
            print(f"  Processing tile {i+1}/{len(tiles)}...", end='\r')

            tile_detections = self.detect_holes_in_tile(
                tile['image'],
                tile['mask'],
                patch_size=48,
                stride=24,
                contamination=0.08
            )

            for det in tile_detections:
                det['bbox']['x'] += tile['x_offset']
                det['bbox']['y'] += tile['y_offset']
                all_detections.append(det)

        print(f"\n  Found {len(all_detections)} candidate holes")

        print("\nStep 4: Merging overlapping detections...")
        merged_detections = self.merge_overlapping_detections(all_detections, iou_threshold=0.3)
        print(f"  Final count: {len(merged_detections)} holes")

        merged_detections = [d for d in merged_detections if d['confidence'] >= min_confidence]
        merged_detections.sort(key=lambda d: d['confidence'], reverse=True)

        return merged_detections

    def merge_overlapping_detections(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping detections using NMS."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)

        merged = []

        for det in detections:
            bbox = det['bbox']
            x1, y1 = bbox['x'], bbox['y']
            x2, y2 = x1 + bbox['w'], y1 + bbox['h']

            overlap = False
            for kept in merged:
                kbbox = kept['bbox']
                kx1, ky1 = kbbox['x'], kbbox['y']
                kx2, ky2 = kx1 + kbbox['w'], ky1 + kbbox['h']

                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih

                area1 = (x2 - x1) * (y2 - y1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                union = area1 + area2 - intersection

                iou = intersection / union if union > 0 else 0

                if iou > iou_threshold:
                    overlap = True
                    break

            if not overlap:
                merged.append(det)

        return merged


def draw_detections(image_path: str, detections: List[Dict], output_path: str = "output_segmented_holes.jpg"):
    """Draw bounding boxes on detected holes."""
    img = cv2.imread(image_path)

    if not detections:
        cv2.putText(img, "No holes detected", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imwrite(output_path, img)
        print(f"Output saved to {output_path}")
        return

    for i, det in enumerate(detections):
        bbox = det['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        conf = det['confidence']
        area = det['area_pixels']

        color = (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        label = f"HOLE #{i+1}"
        conf_label = f"{conf:.2f} ({int(area)}px²)"

        cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.putText(img, conf_label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")


def test_segmented_detection():
    """Test segmented hole detection."""
    test_image = "test_shirt.jpg"

    print("=" * 70)
    print("SEGMENTED GARMENT HOLE DETECTION")
    print("Strategy: Segment → Tile → Detect → Merge")
    print("=" * 70)
    print(f"\nAnalyzing: {test_image}\n")

    detector = SegmentedHoleDetector()

    detections = detector.detect_holes(
        test_image,
        tile_size=512,
        overlap=128,
        min_confidence=0.7
    )

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {len(detections)} hole(s) detected")
    print("=" * 70)

    if detections:
        print("\nDetailed Results (JSON):")
        print(json.dumps(detections, indent=2))

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for i, det in enumerate(detections):
            bbox = det['bbox']
            print(f"\n[HOLE #{i+1}]")
            print(f"  Location: ({bbox['x']}, {bbox['y']})")
            print(f"  Size: {bbox['w']}x{bbox['h']} pixels")
            print(f"  Area: {det['area_pixels']:.0f} px²")
            print(f"  Confidence: {det['confidence']:.2%}")

        draw_detections(test_image, detections)
    else:
        print("\n✓ No holes detected in this garment.")
        draw_detections(test_image, detections)

    print("\n" + "=" * 70)
    print("Debug images saved:")
    print("  - debug_garment_mask.jpg: Segmentation mask")
    print("  - debug_garment_segmented.jpg: Segmented garment")
    print("  - output_segmented_holes.jpg: Final detections")
    print("=" * 70)

    target_x, target_y = 1071, 2555
    print(f"\nChecking for target hole at ({target_x}, {target_y})...")
    for i, det in enumerate(detections):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        if dist < 100:
            print(f"✓ FOUND target hole as detection #{i+1}!")
            print(f"  Distance: {dist:.0f}px")
            print(f"  Confidence: {det['confidence']:.2%}")

    return detections


if __name__ == "__main__":
    test_segmented_detection()