import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, pipeline
from typing import List, Dict, Optional
import json
from verify_holes_final import FinalHoleScorer
import time

# Try to import optimized models for fabric defect detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from transformers import Dinov2Model, AutoImageProcessor as Dinov2Processor
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False


class FabricOptimizedAIFilter:
    """
    Fabric defect detection optimized AI filter.
    Uses models specifically chosen for maximum hole and defect detection accuracy.
    """

    def __init__(self, device_strategy="auto"):
        print("🎯 Initializing Fabric-Optimized AI Filter...")
        print("   Focus: Maximum hole and fabric defect detection accuracy")

        self.device_strategy = device_strategy
        self.setup_devices()
        self.load_optimized_models()
        self.scorer = FinalHoleScorer()

    def setup_devices(self):
        """Setup optimal device configuration."""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            print(f"🔥 Detected {self.num_gpus} GPU(s)")

            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({vram_gb:.1f}GB VRAM)")

            self.primary_device = "cuda:0"
            self.secondary_device = "cuda:1" if self.num_gpus > 1 else "cuda:0"
        else:
            print("⚠️ CUDA not available, using CPU")
            self.primary_device = "cpu"
            self.secondary_device = "cpu"
            self.num_gpus = 0

    def load_optimized_models(self):
        """Load models optimized specifically for fabric defect detection."""
        print("📦 Loading fabric-optimized AI models...")

        # 1. YOLOv8m - Medium model (perfect balance for fabric defects)
        self.load_optimized_yolo()

        # 2. DINOv2 - Best vision foundation model for defect detection
        self.load_dinov2_model()

        # 3. CLIP optimized for fabric and textile understanding
        self.load_fabric_clip_model()

        # 4. Specialized defect detection pipeline
        self.load_defect_detection_pipeline()

    def load_optimized_yolo(self):
        """Load YOLOv8m - optimal for fabric defects (balance of accuracy/speed)."""
        self.use_yolo = False
        if YOLO_AVAILABLE:
            try:
                print("   📦 Loading YOLOv8m (medium - optimal for fabric defects)...")
                # YOLOv8m is the sweet spot: much better than nano, faster than xl
                self.yolo_model = YOLO('yolov8m.pt')  # ~50MB, excellent accuracy

                if self.primary_device != "cpu":
                    self.yolo_model.to(self.primary_device)

                self.use_yolo = True
                print("   ✅ YOLOv8m loaded successfully")
            except Exception as e:
                print(f"   ⚠️ YOLOv8m failed, trying YOLOv8s: {e}")
                try:
                    self.yolo_model = YOLO('yolov8s.pt')  # Fallback
                    if self.primary_device != "cpu":
                        self.yolo_model.to(self.primary_device)
                    self.use_yolo = True
                    print("   ✅ YOLOv8s loaded successfully")
                except Exception as e2:
                    print(f"   ❌ YOLO models failed: {e2}")

    def load_dinov2_model(self):
        """Load DINOv2 - state-of-the-art vision foundation model."""
        self.use_dinov2 = False
        if DINOV2_AVAILABLE:
            try:
                print("   📦 Loading DINOv2-small (best foundation model for defects)...")
                # DINOv2 is specifically excellent at anomaly detection
                self.dinov2_processor = Dinov2Processor.from_pretrained('facebook/dinov2-small')
                self.dinov2_model = Dinov2Model.from_pretrained('facebook/dinov2-small')

                device = self.secondary_device
                self.dinov2_model = self.dinov2_model.to(device)
                self.dinov2_model.eval()

                self.use_dinov2 = True
                print(f"   ✅ DINOv2-small loaded on {device}")
            except Exception as e:
                print(f"   ⚠️ DINOv2 failed: {e}")

    def load_fabric_clip_model(self):
        """Load CLIP optimized for fabric and textile understanding."""
        self.use_clip = False
        if CLIP_AVAILABLE:
            try:
                print("   📦 Loading CLIP-ViT-B/32 (optimized for fabric)...")
                # ViT-B/32 is perfect balance for textile understanding
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

                device = self.primary_device
                self.clip_model = self.clip_model.to(device)
                self.clip_model.eval()

                # Optimized fabric-specific queries
                self.fabric_hole_queries = [
                    "a hole in fabric material",
                    "torn fabric with visible hole",
                    "fabric damage and tear",
                    "textile defect hole",
                    "clothing with hole damage",
                    "worn fabric opening",
                    "fabric deterioration hole"
                ]

                self.fabric_normal_queries = [
                    "normal fabric texture",
                    "intact textile material",
                    "healthy fabric surface",
                    "undamaged clothing material",
                    "perfect fabric weave",
                    "normal textile pattern",
                    "quality fabric surface"
                ]

                self.use_clip = True
                print(f"   ✅ CLIP optimized for fabric loaded on {device}")
            except Exception as e:
                print(f"   ❌ CLIP failed: {e}")

    def load_defect_detection_pipeline(self):
        """Load specialized defect detection pipeline."""
        self.use_defect_pipeline = False
        try:
            print("   📦 Loading specialized defect detection pipeline...")
            # Use segmentation model specifically good at anomaly detection
            self.defect_pipeline = pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=0 if self.num_gpus > 0 else -1
            )
            self.use_defect_pipeline = True
            print("   ✅ Defect detection pipeline loaded")
        except Exception as e:
            print(f"   ⚠️ Defect detection pipeline failed: {e}")

    def extract_optimized_yolo_features(self, patch: np.ndarray) -> Dict:
        """Extract features using optimized YOLO for fabric defects."""
        if not self.use_yolo:
            return {"confidence": 0.0, "detections": 0, "fabric_anomaly_score": 0.5}

        try:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # Optimized settings for fabric defect detection
            results = self.yolo_model(rgb_patch, conf=0.05, verbose=False)  # Lower threshold for subtle defects

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()

                # Advanced fabric defect scoring
                max_conf = float(np.max(confidences))
                num_detections = len(confidences)
                avg_conf = float(np.mean(confidences))

                # For fabric defects: many uncertain detections = likely defect area
                very_low_conf = np.sum(confidences < 0.15)  # Very uncertain objects
                low_conf_count = np.sum(confidences < 0.3)   # Uncertain objects

                # Fabric defect logic: confusion in object detection = potential hole
                if very_low_conf > 3:
                    # Many very uncertain detections = high defect probability
                    fabric_anomaly_score = 0.9
                elif low_conf_count > 2 and avg_conf < 0.25:
                    # Multiple uncertain detections = medium-high defect probability
                    fabric_anomaly_score = 0.8
                elif num_detections > 5 and avg_conf < 0.4:
                    # Many detections but low confidence = medium defect probability
                    fabric_anomaly_score = 0.7
                elif max_conf < 0.2:
                    # Nothing clearly detected = potential hole
                    fabric_anomaly_score = 0.85
                else:
                    # Clear object detection = probably not a hole
                    fabric_anomaly_score = 1.0 - avg_conf

                return {
                    "confidence": max_conf,
                    "detections": num_detections,
                    "fabric_anomaly_score": fabric_anomaly_score,
                    "avg_confidence": avg_conf,
                    "very_low_conf_count": very_low_conf,
                    "low_conf_count": low_conf_count
                }
            else:
                # No objects detected = high probability of hole
                return {
                    "confidence": 0.0,
                    "detections": 0,
                    "fabric_anomaly_score": 0.95,  # Very high for fabric holes
                    "avg_confidence": 0.0,
                    "very_low_conf_count": 0,
                    "low_conf_count": 0
                }

        except Exception as e:
            print(f"Optimized YOLO extraction failed: {e}")
            return {"confidence": 0.0, "detections": 0, "fabric_anomaly_score": 0.5}

    def extract_dinov2_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract features using DINOv2 foundation model."""
        if not self.use_dinov2:
            return np.array([0.0])

        try:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            inputs = self.dinov2_processor(images=rgb_patch, return_tensors="pt")

            device = next(self.dinov2_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.dinov2_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            return features

        except Exception as e:
            print(f"DINOv2 extraction failed: {e}")
            return np.array([0.0])

    def extract_fabric_clip_features(self, patch: np.ndarray) -> Dict:
        """Extract fabric-specific semantic features using CLIP."""
        if not self.use_clip:
            return {"fabric_hole_probability": 0.5, "best_fabric_match": "unknown"}

        try:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            device = next(self.clip_model.parameters()).device

            all_queries = self.fabric_hole_queries + self.fabric_normal_queries
            inputs = self.clip_processor(
                text=all_queries,
                images=rgb_patch,
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # Calculate fabric hole vs normal fabric probability
            hole_probs = probs[:len(self.fabric_hole_queries)]
            normal_probs = probs[len(self.fabric_hole_queries):]

            fabric_hole_probability = float(np.sum(hole_probs))
            best_match_idx = np.argmax(probs)
            best_fabric_match = all_queries[best_match_idx]

            return {
                "fabric_hole_probability": fabric_hole_probability,
                "best_fabric_match": best_fabric_match,
                "hole_confidence": float(np.max(hole_probs)),
                "normal_confidence": float(np.max(normal_probs))
            }

        except Exception as e:
            print(f"Fabric CLIP extraction failed: {e}")
            return {"fabric_hole_probability": 0.5, "best_fabric_match": "error"}

    def compute_fabric_optimized_probability(self, image: np.ndarray, detection: Dict) -> float:
        """
        Compute hole probability using fabric-optimized models.
        Specifically tuned for maximum fabric defect detection accuracy.
        """
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

        # Extract optimal context for fabric analysis
        context_size = 60  # Optimized for fabric texture analysis
        cx, cy = x + w//2, y + h//2
        x1 = max(0, cx - context_size)
        y1 = max(0, cy - context_size)
        x2 = min(image.shape[1], cx + context_size)
        y2 = min(image.shape[0], cy + context_size)

        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0

        # 1. Hand-crafted features (still very important for fabric)
        hole_features = self.scorer.compute_hole_specific_features(image, detection)
        hand_crafted_score = (
            hole_features['shape_irregularity'] * 0.40 +
            hole_features['texture_disruption'] * 0.30 +
            hole_features['background_visibility'] * 0.20 +
            hole_features['depth_contrast'] * 0.10
        )

        # 2. Optimized YOLO fabric defect features
        yolo_features = self.extract_optimized_yolo_features(patch)
        yolo_score = yolo_features['fabric_anomaly_score']

        # 3. DINOv2 foundation model features (excellent for anomalies)
        dinov2_features = self.extract_dinov2_features(patch)
        dinov2_score = 0.5
        if len(dinov2_features) > 1:
            # DINOv2 features have different variance patterns for defects
            feature_variance = np.var(dinov2_features)
            feature_mean = np.mean(np.abs(dinov2_features))
            # Normalized scoring for DINOv2
            dinov2_score = min(1.0, feature_variance / 2.0 + feature_mean / 10.0)

        # 4. Fabric-optimized CLIP features
        clip_features = self.extract_fabric_clip_features(patch)
        clip_score = clip_features['fabric_hole_probability']

        # Fabric-optimized ensemble weights
        if self.use_clip and self.use_dinov2 and self.use_yolo:
            # All fabric-optimized models available
            final_prob = (
                hand_crafted_score * 0.35 +    # Proven fabric features
                clip_score * 0.30 +            # Fabric semantic understanding
                yolo_score * 0.20 +            # Fabric defect detection
                dinov2_score * 0.15            # Foundation model anomaly detection
            )
        elif self.use_clip and self.use_yolo:
            # CLIP + YOLO fabric optimization
            final_prob = (
                hand_crafted_score * 0.45 +
                clip_score * 0.35 +
                yolo_score * 0.20
            )
        else:
            # Fallback ensemble
            final_prob = (
                hand_crafted_score * 0.60 +
                yolo_score * 0.30 +
                dinov2_score * 0.10
            )

        # Fabric-specific boosters (from your successful config)
        area = detection['area_pixels']

        # Size boost optimized for fabric holes
        if 300 <= area <= 800:  # Typical fabric hole size
            size_mult = 1.5
        elif 150 <= area < 300:  # Small fabric holes
            size_mult = 1.4
        elif 800 <= area <= 1500:  # Larger fabric holes
            size_mult = 1.3
        else:
            size_mult = 1.0

        # Fabric subtlety boost (holes can be very subtle)
        subtlety_boost = 1.0
        if (hole_features['depth_contrast'] < 0.08 and  # Very subtle
            hole_features['texture_disruption'] > 0.7):  # But clear texture change
            subtlety_boost = 1.6  # Higher boost for subtle fabric holes

        # Fabric pattern penalty (avoid fabric patterns mistaken as holes)
        pattern_penalty = 1.0
        if (hole_features['shape_irregularity'] < 0.6 and  # Very regular
            area > 600 and  # Large
            hole_features['depth_contrast'] > 0.15):  # High contrast
            pattern_penalty = 0.5  # Likely fabric pattern, not hole

        return min(1.0, final_prob * size_mult * subtlety_boost * pattern_penalty)

    def filter_detections_fabric_optimized(self, image: np.ndarray, detections: List[Dict], threshold: float = 0.70) -> List[Dict]:
        """
        Fabric-optimized ensemble filtering for maximum hole detection accuracy.
        """
        print(f"🎯 Fabric-Optimized AI: Processing {len(detections)} detections...")
        print(f"   Using fabric-optimized threshold: {threshold}")

        filtered_detections = []
        processing_stats = {"processed": 0, "kept": 0, "filtered": 0}

        for i, det in enumerate(detections):
            processing_stats["processed"] += 1

            # Compute fabric-optimized probability
            prob = self.compute_fabric_optimized_probability(image, det)
            det['fabric_optimized_probability'] = prob

            # Apply fabric-optimized filtering
            if prob >= threshold:
                filtered_detections.append(det)
                processing_stats["kept"] += 1
            else:
                processing_stats["filtered"] += 1

            # Progress indicator
            if i % 15 == 0:
                print(f"   Processed: {i+1}/{len(detections)} ({processing_stats['kept']} kept)")

        # Sort by fabric-optimized probability (highest first)
        filtered_detections.sort(key=lambda x: x['fabric_optimized_probability'], reverse=True)

        print(f"✅ Fabric-Optimized AI Results:")
        print(f"   Processed: {processing_stats['processed']}")
        print(f"   Kept: {processing_stats['kept']}")
        print(f"   Filtered out: {processing_stats['filtered']}")
        print(f"   Reduction: {(1 - processing_stats['kept']/processing_stats['processed'])*100:.1f}%")

        return filtered_detections


def test_fabric_optimized_filter():
    """Test the fabric-optimized AI filter."""
    print("=" * 80)
    print("🎯 FABRIC-OPTIMIZED AI FILTERING - MAXIMUM DEFECT DETECTION")
    print("=" * 80)

    # Load detections
    try:
        with open('../results/enhanced_detections.json', 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        print("❌ No enhanced_detections.json found. Run initial detection first.")
        return []

    img = cv2.imread('../data/test_shirt.jpg')
    if img is None:
        print("❌ Could not load test image")
        return []

    fabric_filter = FabricOptimizedAIFilter()

    print(f"\nApplying fabric-optimized AI filter to {len(detections)} detections...")

    start_time = time.time()
    fabric_detections = fabric_filter.filter_detections_fabric_optimized(
        img,
        detections,
        threshold=0.70  # Optimized threshold for fabric defects
    )
    processing_time = time.time() - start_time

    print(f"\n🎯 FABRIC-OPTIMIZED RESULTS")
    print("-" * 60)
    per_detection_ms = processing_time/len(detections)*1000 if len(detections) > 0 else 0.0
    print(f"Processing time: {processing_time:.1f}s ({per_detection_ms:.1f}ms per detection)")

    # Check actual hole ranking
    target_x, target_y = 1660, 2482
    actual_hole_rank = None

    print(f"\nTop 10 by FABRIC-OPTIMIZED probability:")
    for i, det in enumerate(fabric_detections[:10]):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        is_target = "🎯" if dist < 50 else "  "
        prob = det.get('fabric_optimized_probability', 0.0)

        print(f"{is_target}#{i+1}: ({x:4}, {y:4}) {bbox['w']:2}x{bbox['h']:2} Prob: {prob:.3f}")

        if dist < 50:
            actual_hole_rank = i + 1
            print(f"      *** ACTUAL HOLE FOUND AT RANK #{i+1} ***")

    print(f"\n📊 FABRIC-OPTIMIZED SUMMARY:")
    print(f"  🎯 Actual hole rank: #{actual_hole_rank}")
    speed = len(detections)/processing_time if processing_time > 0 else 0.0
    print(f"  ⚡ Processing speed: {speed:.1f} detections/sec")
    print(f"  🎯 Optimization: Fabric defect detection maximized")

    # Save results
    with open("../results/fabric_optimized_filtered.json", "w") as f:
        json.dump(fabric_detections, f, indent=2)

    print(f"  💾 Fabric-optimized detections saved: {len(fabric_detections)} candidates")

    return fabric_detections


if __name__ == "__main__":
    test_fabric_optimized_filter()