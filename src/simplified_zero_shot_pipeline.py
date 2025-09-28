import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Optional, Tuple
import json
import time
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*")


class SimplifiedZeroShotPipeline:
    """
    Simplified zero-shot fabric defect detection pipeline.

    Uses only stable components:
    1. WinCLIP → Anomaly heatmap (stable CLIP model)
    2. Simple masking → Convert peaks to masks
    3. Heuristic grounding → Shape-based confirmation
    4. Multi-modal confirmation → Spatial overlap logic
    """

    def __init__(self, device_strategy="auto"):
        print("🚀 Initializing Simplified Zero-Shot Pipeline...")
        print("   Components: WinCLIP + Simple Masking + Heuristic Grounding")

        self.setup_devices()
        self.load_stable_models()
        self.setup_fabric_prompts()

    def setup_devices(self):
        """Setup device configuration."""
        if torch.cuda.is_available():
            self.device = "cuda:0"
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            print(f"🔥 Using GPU: {props.name} ({vram_gb:.1f}GB VRAM)")
        else:
            self.device = "cpu"
            print("⚠️ Using CPU")

    def load_stable_models(self):
        """Load only stable, proven models."""
        print("📦 Loading stable models...")

        try:
            print("   📦 Loading CLIP for WinCLIP (stable version)...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()

            # WinCLIP parameters
            self.patch_size = 32
            self.stride = 16

            print("   ✅ Stable CLIP model loaded")
        except Exception as e:
            print(f"   ❌ CLIP loading failed: {e}")
            raise

    def setup_fabric_prompts(self):
        """Setup fabric-specific prompts."""
        print("   📝 Setting up fabric prompts...")

        # Core fabric anomaly prompts
        self.anomaly_prompts = [
            "hole in fabric",
            "tear in material",
            "puncture in textile",
            "run in knit",
            "damaged stitch",
            "fabric defect",
            "worn fabric",
            "textile damage"
        ]

        self.normal_prompts = [
            "normal fabric",
            "intact textile",
            "perfect material",
            "undamaged cloth",
            "healthy fabric",
            "good textile"
        ]

        print(f"   ✅ Setup {len(self.anomaly_prompts)} anomaly prompts")

    def generate_winclip_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Generate anomaly heatmap using stable WinCLIP."""
        print("🎯 Generating WinCLIP heatmap...")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]

        # Create heatmap grid
        grid_h = max(1, h // self.stride)
        grid_w = max(1, w // self.stride)
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

        all_prompts = self.anomaly_prompts + self.normal_prompts

        # Process patches
        patches_processed = 0
        for i in range(grid_h):
            for j in range(grid_w):
                y = min(i * self.stride, h - self.patch_size)
                x = min(j * self.stride, w - self.patch_size)

                # Extract patch
                patch = rgb_image[y:y+self.patch_size, x:x+self.patch_size]

                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))

                try:
                    inputs = self.clip_processor(
                        text=all_prompts,
                        images=patch,
                        return_tensors="pt",
                        padding=True
                    )

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.clip_model(**inputs)
                        logits = outputs.logits_per_image
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                    # Compute anomaly score
                    anomaly_probs = probs[:len(self.anomaly_prompts)]
                    normal_probs = probs[len(self.anomaly_prompts):]

                    anomaly_score = np.mean(anomaly_probs)
                    normal_score = np.mean(normal_probs)

                    final_score = anomaly_score / (anomaly_score + normal_score + 1e-8)
                    heatmap[i, j] = final_score

                    patches_processed += 1

                except Exception as e:
                    print(f"Patch {i},{j} failed: {e}")
                    heatmap[i, j] = 0.0

        # Resize to original image size
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        print(f"   ✅ Processed {patches_processed} patches, max score: {np.max(heatmap_resized):.3f}")
        return heatmap_resized

    def heatmap_to_masks(self, heatmap: np.ndarray, threshold: float = 0.6) -> List[np.ndarray]:
        """Convert heatmap peaks to masks."""
        print(f"🎭 Converting heatmap to masks (threshold: {threshold})...")

        # Threshold heatmap
        binary_map = (heatmap > threshold).astype(np.uint8)

        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_map)

        masks = []
        for label_id in range(1, num_labels):
            mask = (labels == label_id).astype(np.uint8)
            area = np.sum(mask)

            # Filter by size
            if 30 <= area <= 5000:  # Reasonable hole sizes
                masks.append(mask)

        print(f"   ✅ Generated {len(masks)} masks")
        return masks

    def heuristic_grounding(self, image: np.ndarray, masks: List[np.ndarray]) -> List[Dict]:
        """Apply heuristic grounding based on shape properties."""
        print("🔍 Applying heuristic grounding...")

        detections = []

        for i, mask in enumerate(masks):
            try:
                # Get bounding box
                coords = np.where(mask > 0)
                if len(coords[0]) == 0:
                    continue

                y1, y2 = np.min(coords[0]), np.max(coords[0])
                x1, x2 = np.min(coords[1]), np.max(coords[1])

                # Calculate properties
                area = np.sum(mask > 0)
                width = x2 - x1 + 1
                height = y2 - y1 + 1
                aspect_ratio = width / max(1, height)

                # Heuristic scoring
                score = 0.4  # Base score

                # Size scoring (typical fabric holes)
                if 50 <= area <= 2000:
                    score += 0.2
                elif 2000 < area <= 5000:
                    score += 0.1

                # Aspect ratio (holes tend to be roughly round)
                if 0.4 <= aspect_ratio <= 2.5:
                    score += 0.2

                # Compactness (holes are usually compact)
                try:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        perimeter = cv2.arcLength(contours[0], True)
                        if perimeter > 0:
                            compactness = 4 * np.pi * area / (perimeter * perimeter)
                            if compactness > 0.2:
                                score += 0.15
                            if compactness > 0.4:
                                score += 0.05
                except:
                    pass

                # Solidity (filled vs outline)
                try:
                    hull = cv2.convexHull(contours[0])
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.7:  # More solid = more hole-like
                            score += 0.1
                except:
                    pass

                detections.append({
                    'bbox': {'x': int(x1), 'y': int(y1), 'w': int(width), 'h': int(height)},
                    'mask': mask,
                    'grounding_score': min(1.0, score),
                    'heuristic_properties': {
                        'area': int(area),
                        'aspect_ratio': float(aspect_ratio),
                        'compactness': float(compactness) if 'compactness' in locals() else 0.0
                    }
                })

            except Exception as e:
                print(f"Heuristic grounding failed for mask {i}: {e}")
                continue

        print(f"   ✅ Grounding complete: {len(detections)} candidates")
        return detections

    def multi_modal_confirmation(self, detections: List[Dict], heatmap: np.ndarray,
                                winclip_threshold: float = 0.6,
                                grounding_threshold: float = 0.5) -> List[Dict]:
        """Apply multi-modal confirmation logic."""
        print("📍 Applying multi-modal confirmation...")

        confirmed_detections = []

        for det in detections:
            bbox = det['bbox']
            grounding_score = det.get('grounding_score', 0.0)

            # Get WinCLIP score from heatmap
            center_x = bbox['x'] + bbox['w'] // 2
            center_y = bbox['y'] + bbox['h'] // 2

            if 0 <= center_y < heatmap.shape[0] and 0 <= center_x < heatmap.shape[1]:
                winclip_score = float(heatmap[center_y, center_x])
            else:
                winclip_score = 0.0

            # Multi-modal confirmation logic
            confirmed = False
            confidence_reason = ""
            final_confidence = 0.0

            if winclip_score >= 0.8 and grounding_score >= 0.7:
                # Strong agreement from both
                confirmed = True
                confidence_reason = "strong_consensus"
                final_confidence = 0.9

            elif winclip_score >= 0.75:
                # Very high WinCLIP score
                confirmed = True
                confidence_reason = "high_winclip"
                final_confidence = 0.85

            elif winclip_score >= winclip_threshold and grounding_score >= grounding_threshold:
                # Both above threshold
                confirmed = True
                confidence_reason = "moderate_consensus"
                final_confidence = 0.75

            elif winclip_score >= 0.7 or grounding_score >= 0.8:
                # One very strong signal
                confirmed = True
                confidence_reason = "single_strong_signal"
                final_confidence = 0.7

            if confirmed:
                det['winclip_score'] = winclip_score
                det['final_confidence'] = final_confidence
                det['confidence_reason'] = confidence_reason
                det['confirmed'] = True
                confirmed_detections.append(det)

        print(f"   ✅ Confirmed {len(confirmed_detections)}/{len(detections)} detections")
        return confirmed_detections

    def run_simplified_pipeline(self, image: np.ndarray,
                               winclip_threshold: float = 0.6,
                               grounding_threshold: float = 0.5) -> List[Dict]:
        """Run the simplified zero-shot pipeline."""
        print("🚀 Running Simplified Zero-Shot Pipeline...")
        start_time = time.time()

        # Step 1: WinCLIP heatmap
        heatmap = self.generate_winclip_heatmap(image)

        # Step 2: Convert to masks
        masks = self.heatmap_to_masks(heatmap, threshold=winclip_threshold)

        if not masks:
            print("   ℹ️ No candidate regions found")
            return []

        # Step 3: Heuristic grounding
        detections = self.heuristic_grounding(image, masks)

        # Step 4: Multi-modal confirmation
        confirmed_detections = self.multi_modal_confirmation(
            detections, heatmap, winclip_threshold, grounding_threshold
        )

        processing_time = time.time() - start_time

        print(f"🎯 Simplified Pipeline Complete!")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Confirmed defects: {len(confirmed_detections)}")

        return confirmed_detections


def test_simplified_pipeline():
    """Test the simplified pipeline."""
    print("=" * 70)
    print("🚀 SIMPLIFIED ZERO-SHOT FABRIC DEFECT DETECTION")
    print("=" * 70)

    img = cv2.imread('../data/test_shirt.jpg')
    if img is None:
        print("❌ Could not load test image")
        return []

    pipeline = SimplifiedZeroShotPipeline()

    detections = pipeline.run_simplified_pipeline(
        img,
        winclip_threshold=0.6,
        grounding_threshold=0.5
    )

    # Save results
    with open("../results/simplified_zero_shot.json", "w") as f:
        json_detections = []
        for det in detections:
            json_det = det.copy()
            if 'mask' in json_det:
                del json_det['mask']
            json_detections.append(json_det)
        json.dump(json_detections, f, indent=2)

    print(f"\n📊 RESULTS:")
    for i, det in enumerate(detections[:5]):
        bbox = det['bbox']
        conf = det.get('final_confidence', 0.0)
        reason = det.get('confidence_reason', 'unknown')
        winclip = det.get('winclip_score', 0.0)

        print(f"  #{i+1}: ({bbox['x']}, {bbox['y']}) {bbox['w']}x{bbox['h']} "
              f"Conf: {conf:.3f} WinCLIP: {winclip:.3f} ({reason})")

    return detections


if __name__ == "__main__":
    test_simplified_pipeline()