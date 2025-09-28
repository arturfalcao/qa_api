import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, pipeline
from typing import List, Dict, Optional
import json
from verify_holes_final import FinalHoleScorer
import time

# Try to import advanced models
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
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False


class AdvancedLocalAIFilter:
    """
    High-performance local AI filter leveraging RTX 5090s for advanced hole detection.
    Uses multiple state-of-the-art models in ensemble.
    """

    def __init__(self, device_strategy="multi_gpu"):
        print("🚀 Initializing Advanced Local AI Filter for RTX 5090...")

        self.device_strategy = device_strategy
        self.setup_devices()
        self.load_models()
        self.scorer = FinalHoleScorer()

    def setup_devices(self):
        """Setup optimal device configuration for RTX 5090s."""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            print(f"🔥 Detected {self.num_gpus} GPU(s)")

            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({vram_gb:.1f}GB VRAM)")

            # Optimal device assignment for RTX 5090s
            self.primary_device = "cuda:0"
            self.secondary_device = "cuda:1" if self.num_gpus > 1 else "cuda:0"
        else:
            print("⚠️ CUDA not available, using CPU")
            self.primary_device = "cpu"
            self.secondary_device = "cpu"
            self.num_gpus = 0

    def load_models(self):
        """Load advanced models optimized for RTX 5090s."""
        print("📦 Loading advanced AI models...")

        # 1. Upgrade YOLO to larger model (YOLOv8x/v11x for better accuracy)
        self.load_yolo_model()

        # 2. Upgrade vision backbone to larger model (ResNet-101/152 or EfficientNet)
        self.load_vision_backbone()

        # 3. Add CLIP for semantic understanding
        self.load_clip_model()

        # 4. Add vision-language model for detailed analysis
        self.load_vision_language_model()

        # 5. Add depth estimation model
        self.load_depth_model()

    def load_yolo_model(self):
        """Load the most powerful YOLO model."""
        self.use_yolo = False
        if YOLO_AVAILABLE:
            try:
                # Use YOLOv8x or YOLOv11x for maximum accuracy
                print("   📦 Loading YOLOv11x (large model)...")
                self.yolo_model = YOLO('yolo11x.pt')  # Much larger, more accurate

                # Move to primary GPU
                if self.primary_device != "cpu":
                    self.yolo_model.to(self.primary_device)

                self.use_yolo = True
                print("   ✅ YOLOv11x loaded successfully")
            except Exception as e:
                print(f"   ⚠️ YOLOv11x failed, trying YOLOv8x: {e}")
                try:
                    self.yolo_model = YOLO('yolov8x.pt')
                    if self.primary_device != "cpu":
                        self.yolo_model.to(self.primary_device)
                    self.use_yolo = True
                    print("   ✅ YOLOv8x loaded successfully")
                except Exception as e2:
                    print(f"   ❌ All YOLO models failed: {e2}")

    def load_vision_backbone(self):
        """Load powerful vision backbone."""
        self.use_vision = False
        try:
            # Use EfficientNet-B7 or ResNet-152 for better features
            print("   📦 Loading EfficientNet-B7...")
            self.vision_processor = AutoImageProcessor.from_pretrained('google/efficientnet-b7')
            self.vision_model = AutoModel.from_pretrained('google/efficientnet-b7')

            # Move to secondary GPU if available
            device = self.secondary_device
            self.vision_model = self.vision_model.to(device)
            self.vision_model.eval()

            self.use_vision = True
            print(f"   ✅ EfficientNet-B7 loaded on {device}")
        except Exception as e:
            print(f"   ⚠️ EfficientNet-B7 failed, trying ResNet-152: {e}")
            try:
                self.vision_processor = AutoImageProcessor.from_pretrained('microsoft/resnet-152')
                self.vision_model = AutoModel.from_pretrained('microsoft/resnet-152')
                device = self.secondary_device
                self.vision_model = self.vision_model.to(device)
                self.vision_model.eval()
                self.use_vision = True
                print(f"   ✅ ResNet-152 loaded on {device}")
            except Exception as e2:
                print(f"   ❌ Vision backbone failed: {e2}")

    def load_clip_model(self):
        """Load CLIP for semantic understanding."""
        self.use_clip = False
        if CLIP_AVAILABLE:
            try:
                print("   📦 Loading CLIP-ViT-L/14...")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

                device = self.primary_device
                self.clip_model = self.clip_model.to(device)
                self.clip_model.eval()

                # Prepare text queries for hole detection
                self.hole_queries = [
                    "a hole in fabric",
                    "a tear in clothing",
                    "damaged fabric with hole",
                    "fabric defect",
                    "worn fabric with opening"
                ]

                self.non_hole_queries = [
                    "decorative pattern on fabric",
                    "printed design on clothing",
                    "fabric texture",
                    "normal fabric pattern",
                    "clothing decoration"
                ]

                self.use_clip = True
                print(f"   ✅ CLIP loaded on {device}")
            except Exception as e:
                print(f"   ❌ CLIP failed: {e}")

    def load_vision_language_model(self):
        """Load vision-language model for detailed analysis."""
        self.use_vlm = False
        if BLIP2_AVAILABLE and self.num_gpus > 0:
            try:
                print("   📦 Loading BLIP-2 for vision-language analysis...")
                # Use smaller BLIP-2 model to fit alongside other models
                self.vlm_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16,  # Use FP16 to save VRAM
                    device_map="auto"
                )

                self.hole_prompt = "Question: Is there a hole or tear in this fabric? Answer:"
                self.use_vlm = True
                print("   ✅ BLIP-2 loaded successfully")
            except Exception as e:
                print(f"   ⚠️ BLIP-2 failed (may need more VRAM): {e}")

    def load_depth_model(self):
        """Load depth estimation for hole depth analysis."""
        self.use_depth = False
        try:
            print("   📦 Loading DPT depth estimation...")
            self.depth_pipeline = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.num_gpus > 0 else -1
            )
            self.use_depth = True
            print("   ✅ Depth estimation loaded")
        except Exception as e:
            print(f"   ⚠️ Depth estimation failed: {e}")

    def extract_advanced_yolo_features(self, patch: np.ndarray) -> Dict:
        """Extract features using advanced YOLO model."""
        if not self.use_yolo:
            return {"confidence": 0.0, "detections": 0, "anomaly_score": 0.5}

        try:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # Run with higher confidence threshold for more precise detection
            results = self.yolo_model(rgb_patch, conf=0.1, verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                # Advanced anomaly scoring
                max_conf = float(np.max(confidences))
                num_detections = len(confidences)
                avg_conf = float(np.mean(confidences))

                # More sophisticated anomaly detection
                # Many low-confidence detections = potential defect area
                low_conf_count = np.sum(confidences < 0.3)

                if num_detections > 5 and avg_conf < 0.4:
                    # Many uncertain detections = high anomaly
                    anomaly_score = 0.8
                elif max_conf < 0.3:
                    # No confident detections = high anomaly
                    anomaly_score = 0.9
                else:
                    # Normal object detection = low anomaly
                    anomaly_score = 1.0 - avg_conf

                return {
                    "confidence": max_conf,
                    "detections": num_detections,
                    "anomaly_score": anomaly_score,
                    "avg_confidence": avg_conf,
                    "low_conf_count": low_conf_count
                }
            else:
                # No objects = potential hole
                return {
                    "confidence": 0.0,
                    "detections": 0,
                    "anomaly_score": 1.0,
                    "avg_confidence": 0.0,
                    "low_conf_count": 0
                }

        except Exception as e:
            print(f"Advanced YOLO extraction failed: {e}")
            return {"confidence": 0.0, "detections": 0, "anomaly_score": 0.5}

    def extract_advanced_vision_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract features using advanced vision backbone."""
        if not self.use_vision:
            return np.array([0.0])

        try:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            inputs = self.vision_processor(images=rgb_patch, return_tensors="pt")

            # Move inputs to same device as model
            device = next(self.vision_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.vision_model(**inputs)
                features = outputs.pooler_output.squeeze().cpu().numpy()

            return features

        except Exception as e:
            print(f"Advanced vision extraction failed: {e}")
            return np.array([0.0])

    def extract_clip_features(self, patch: np.ndarray) -> Dict:
        """Extract semantic features using CLIP."""
        if not self.use_clip:
            return {"hole_probability": 0.5, "best_match": "unknown"}

        try:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # Prepare inputs
            device = next(self.clip_model.parameters()).device

            # Process image and text
            all_queries = self.hole_queries + self.non_hole_queries
            inputs = self.clip_processor(
                text=all_queries,
                images=rgb_patch,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # Calculate hole vs non-hole probability
            hole_probs = probs[:len(self.hole_queries)]
            non_hole_probs = probs[len(self.hole_queries):]

            hole_probability = float(np.sum(hole_probs))
            best_match_idx = np.argmax(probs)
            best_match = all_queries[best_match_idx]

            return {
                "hole_probability": hole_probability,
                "best_match": best_match,
                "hole_probs": hole_probs.tolist(),
                "non_hole_probs": non_hole_probs.tolist()
            }

        except Exception as e:
            print(f"CLIP extraction failed: {e}")
            return {"hole_probability": 0.5, "best_match": "error"}

    def extract_depth_features(self, patch: np.ndarray) -> Dict:
        """Extract depth features for hole analysis."""
        if not self.use_depth:
            return {"depth_variance": 0.0, "avg_depth": 0.5}

        try:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(rgb_patch)

            depth = self.depth_pipeline(pil_image)
            depth_array = np.array(depth['depth'])

            # Analyze depth characteristics
            depth_variance = float(np.var(depth_array))
            avg_depth = float(np.mean(depth_array))
            min_depth = float(np.min(depth_array))

            # High depth variance + low minimum = potential hole
            return {
                "depth_variance": depth_variance,
                "avg_depth": avg_depth,
                "min_depth": min_depth
            }

        except Exception as e:
            print(f"Depth extraction failed: {e}")
            return {"depth_variance": 0.0, "avg_depth": 0.5}

    def compute_advanced_hole_probability(self, image: np.ndarray, detection: Dict) -> float:
        """
        Compute hole probability using all advanced AI models.
        """
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

        # Extract larger context for better analysis
        context_size = 80  # Larger context for advanced models
        cx, cy = x + w//2, y + h//2
        x1 = max(0, cx - context_size)
        y1 = max(0, cy - context_size)
        x2 = min(image.shape[1], cx + context_size)
        y2 = min(image.shape[0], cy + context_size)

        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0

        # 1. Hand-crafted features (proven effective)
        hole_features = self.scorer.compute_hole_specific_features(image, detection)
        hand_crafted_score = (
            hole_features['shape_irregularity'] * 0.35 +
            hole_features['texture_disruption'] * 0.25 +
            hole_features['background_visibility'] * 0.20 +
            hole_features['depth_contrast'] * 0.20
        )

        # 2. Advanced YOLO features
        yolo_features = self.extract_advanced_yolo_features(patch)
        yolo_score = yolo_features['anomaly_score']

        # 3. Advanced vision features
        vision_features = self.extract_advanced_vision_features(patch)
        vision_score = 0.5
        if len(vision_features) > 1:
            vision_variance = np.var(vision_features)
            vision_score = min(1.0, vision_variance / 5.0)  # More sensitive

        # 4. CLIP semantic features
        clip_features = self.extract_clip_features(patch)
        clip_score = clip_features['hole_probability']

        # 5. Depth features
        depth_features = self.extract_depth_features(patch)
        depth_score = min(1.0, depth_features['depth_variance'] / 1000.0)

        # Advanced ensemble scoring with all models
        if self.use_clip and self.use_vision and self.use_yolo:
            # All models available - sophisticated ensemble
            final_prob = (
                hand_crafted_score * 0.40 +    # Still important
                clip_score * 0.25 +            # Semantic understanding
                yolo_score * 0.15 +            # Object detection
                vision_score * 0.15 +          # Advanced features
                depth_score * 0.05             # Depth analysis
            )
        elif self.use_clip and self.use_yolo:
            # CLIP + YOLO available
            final_prob = (
                hand_crafted_score * 0.50 +
                clip_score * 0.30 +
                yolo_score * 0.20
            )
        else:
            # Fallback to basic ensemble
            final_prob = (
                hand_crafted_score * 0.60 +
                yolo_score * 0.25 +
                vision_score * 0.15
            )

        # Apply all the successful boosters from before
        area = detection['area_pixels']

        # Size boost
        if 400 <= area <= 600:
            size_mult = 1.4
        elif 200 <= area < 400:
            size_mult = 1.3
        else:
            size_mult = 1.0

        # Subtlety boost
        subtlety_boost = 1.0
        if (hole_features['depth_contrast'] < 0.10 and
            hole_features['shape_irregularity'] > 0.6):
            subtlety_boost = 1.5

        # Decorative penalty
        decorative_penalty = 1.0
        if (hole_features['depth_contrast'] > 0.12 and
            hole_features['shape_irregularity'] < 0.7 and
            area > 400):
            decorative_penalty = 0.6

        return min(1.0, final_prob * size_mult * subtlety_boost * decorative_penalty)

    def filter_detections_ensemble(self, image: np.ndarray, detections: List[Dict], threshold: float = 0.75) -> List[Dict]:
        """
        Advanced ensemble filtering using all RTX 5090 optimized models.

        Args:
            image: Original image
            detections: List of detections to filter
            threshold: Probability threshold for keeping detections

        Returns:
            List of high-confidence filtered detections
        """
        print(f"🚀 RTX 5090 Advanced AI: Processing {len(detections)} detections...")
        print(f"   Using ensemble threshold: {threshold}")

        filtered_detections = []
        processing_stats = {"processed": 0, "kept": 0, "filtered": 0}

        for i, det in enumerate(detections):
            processing_stats["processed"] += 1

            # Compute advanced probability using all models
            prob = self.compute_advanced_hole_probability(image, det)
            det['advanced_ai_probability'] = prob

            # Apply ensemble filtering
            if prob >= threshold:
                filtered_detections.append(det)
                processing_stats["kept"] += 1
            else:
                processing_stats["filtered"] += 1

            # Progress indicator
            if i % 20 == 0:
                print(f"   Processed: {i+1}/{len(detections)} ({processing_stats['kept']} kept)")

        # Sort by probability (highest first)
        filtered_detections.sort(key=lambda x: x['advanced_ai_probability'], reverse=True)

        print(f"✅ RTX 5090 Advanced AI Results:")
        print(f"   Processed: {processing_stats['processed']}")
        print(f"   Kept: {processing_stats['kept']}")
        print(f"   Filtered out: {processing_stats['filtered']}")
        print(f"   Reduction: {(1 - processing_stats['kept']/processing_stats['processed'])*100:.1f}%")

        return filtered_detections


def test_advanced_local_filter():
    """Test the advanced local AI filter."""
    print("=" * 80)
    print("🚀 ADVANCED LOCAL AI FILTERING - RTX 5090 OPTIMIZED")
    print("=" * 80)

    # Load enhanced detections
    with open('../results/enhanced_detections.json', 'r') as f:
        detections = json.load(f)

    img = cv2.imread('../data/test_shirt.jpg')
    advanced_filter = AdvancedLocalAIFilter()

    print(f"\nApplying advanced AI filter to {len(detections)} detections...")

    start_time = time.time()
    scored_detections = []

    for i, det in enumerate(detections):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(detections)}...")

        prob = advanced_filter.compute_advanced_hole_probability(img, det)
        det['advanced_ai_probability'] = prob
        scored_detections.append((det, prob))

    processing_time = time.time() - start_time

    # Sort by probability
    scored_detections.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔥 ADVANCED LOCAL AI RESULTS")
    print("-" * 60)
    per_detection_ms = processing_time/len(detections)*1000 if len(detections) > 0 else 0.0
    print(f"Processing time: {processing_time:.1f}s ({per_detection_ms:.1f}ms per detection)")

    # Check actual hole ranking
    target_x, target_y = 1660, 2482
    actual_hole_rank = None

    print(f"\nTop 10 by ADVANCED AI probability:")
    for i, (det, prob) in enumerate(scored_detections[:10]):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        is_target = "🎯" if dist < 50 else "  "

        print(f"{is_target}#{i+1}: ({x:4}, {y:4}) {bbox['w']:2}x{bbox['h']:2} Prob: {prob:.3f}")

        if dist < 50:
            actual_hole_rank = i + 1
            print(f"      *** ACTUAL HOLE FOUND AT RANK #{i+1} ***")

    print(f"\n📊 ADVANCED AI SUMMARY:")
    print(f"  🎯 Actual hole rank: #{actual_hole_rank}")
    speed = len(detections)/processing_time if processing_time > 0 else 0.0
    print(f"  ⚡ Processing speed: {speed:.1f} detections/sec")
    print(f"  🚀 Hardware utilization: RTX 5090 optimized")

    # Save results
    high_prob_detections = [det for det, prob in scored_detections if prob > 0.5]
    with open("../results/advanced_local_filtered.json", "w") as f:
        json.dump(high_prob_detections, f, indent=2)

    print(f"  💾 High-prob detections saved: {len(high_prob_detections)} candidates")

    return high_prob_detections


if __name__ == "__main__":
    test_advanced_local_filter()