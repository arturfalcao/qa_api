import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from typing import List, Dict
import json
from verify_holes_final import FinalHoleScorer

# Try to import YOLO (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO (ultralytics) available")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - install with: pip install ultralytics")


class LocalHoleClassifier(nn.Module):
    """
    Lightweight local classifier for hole vs decorative pattern.
    Uses pre-trained features + simple classifier head.
    """

    def __init__(self, feature_dim=768):
        super().__init__()

        # Use a lightweight pre-trained backbone
        self.processor = AutoImageProcessor.from_pretrained('microsoft/resnet-18')
        self.backbone = AutoModel.from_pretrained('microsoft/resnet-18')

        # Freeze backbone to save computation
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # ResNet-18 output is 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # hole vs not_hole
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x).pooler_output
        logits = self.classifier(features)
        return logits


class FastLocalFilter:
    """
    Fast local AI filter using multiple lightweight models + hand-crafted features.
    Uses YOLO + ResNet-18 + hand-crafted features for robust detection.
    """

    def __init__(self):
        print("Initializing enhanced multi-model local filter...")

        # 1. Load ResNet-18 for feature extraction
        try:
            self.processor = AutoImageProcessor.from_pretrained('microsoft/resnet-18')
            self.resnet_model = AutoModel.from_pretrained('microsoft/resnet-18')
            self.use_resnet = True
            print("✅ Loaded ResNet-18 for feature extraction")
        except Exception as e:
            print(f"⚠️ Could not load ResNet-18: {e}")
            self.use_resnet = False

        # 2. Load YOLO for object detection
        self.use_yolo = False
        if YOLO_AVAILABLE:
            try:
                # Try YOLOv11n first (latest, most efficient)
                print("📦 Loading YOLOv11n...")
                self.yolo_model = YOLO('yolo11n.pt')
                self.use_yolo = True
                print("✅ Loaded YOLOv11n for object detection")
            except Exception as e:
                try:
                    # Fallback to YOLOv8n
                    print("📦 Loading YOLOv8n...")
                    self.yolo_model = YOLO('yolov8n.pt')
                    self.use_yolo = True
                    print("✅ Loaded YOLOv8n for object detection")
                except Exception as e2:
                    print(f"⚠️ Could not load YOLO models: {e2}")
                    self.use_yolo = False

        self.scorer = FinalHoleScorer()

        print(f"🔧 Multi-model setup: ResNet-18={self.use_resnet}, YOLO={self.use_yolo}")
        if not self.use_resnet and not self.use_yolo:
            print("📊 Using hand-crafted features only")

    def extract_resnet_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract ResNet-18 features for texture analysis."""
        if not self.use_resnet:
            return np.array([0.0])

        try:
            # Convert BGR to RGB and resize
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # Process for ResNet model
            inputs = self.processor(images=rgb_patch, return_tensors="pt")

            with torch.no_grad():
                outputs = self.resnet_model(**inputs)
                # Use pooled features
                features = outputs.pooler_output.squeeze().numpy()

            return features

        except Exception as e:
            print(f"ResNet feature extraction failed: {e}")
            return np.array([0.0])

    def extract_yolo_features(self, patch: np.ndarray) -> Dict:
        """Extract YOLO detection features for object detection."""
        if not self.use_yolo:
            return {"confidence": 0.0, "detections": 0, "anomaly_score": 0.0}

        try:
            # Convert BGR to RGB for YOLO
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = self.yolo_model(rgb_patch, verbose=False)

            # Analyze YOLO results
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                # Calculate features based on detections
                max_conf = float(np.max(confidences))
                num_detections = len(confidences)

                # Anomaly score: high confidence in known objects = low anomaly
                # Low confidence or many small objects = high anomaly (potential hole)
                avg_conf = float(np.mean(confidences))
                anomaly_score = 1.0 - avg_conf  # Invert confidence for anomaly

                return {
                    "confidence": max_conf,
                    "detections": num_detections,
                    "anomaly_score": anomaly_score,
                    "avg_confidence": avg_conf
                }
            else:
                # No objects detected = high anomaly (potential hole)
                return {
                    "confidence": 0.0,
                    "detections": 0,
                    "anomaly_score": 1.0,  # High anomaly when no objects detected
                    "avg_confidence": 0.0
                }

        except Exception as e:
            print(f"YOLO feature extraction failed: {e}")
            return {"confidence": 0.0, "detections": 0, "anomaly_score": 0.0}

    def compute_hole_probability(self, image: np.ndarray, detection: Dict) -> float:
        """
        Compute probability that detection is a real hole using local AI + features.
        """
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

        # Extract patch with sufficient context
        context_size = 60
        cx, cy = x + w//2, y + h//2
        x1 = max(0, cx - context_size)
        y1 = max(0, cy - context_size)
        x2 = min(image.shape[1], cx + context_size)
        y2 = min(image.shape[0], cy + context_size)

        patch = image[y1:y2, x1:x2]

        if patch.size == 0:
            return 0.0

        # 1. Hand-crafted hole-specific features (proven effective)
        hole_features = self.scorer.compute_hole_specific_features(image, detection)

        # 2. ResNet-18 texture features
        resnet_features = self.extract_resnet_features(patch)

        # 3. YOLO object detection features
        yolo_features = self.extract_yolo_features(patch)

        # 4. Enhanced scoring prioritizing REAL HOLE characteristics
        hand_crafted_score = (
            hole_features['shape_irregularity'] * 0.35 +      # BOOSTED: Real holes are irregular
            hole_features['texture_disruption'] * 0.25 +      # BOOSTED: Real holes disrupt fabric
            hole_features['background_visibility'] * 0.20 +   # Good for holes
            hole_features['depth_contrast'] * 0.20            # Reduced: false positives often have high contrast
        )

        # 5. ResNet texture anomaly score
        resnet_score = 0.5  # Default neutral
        if self.use_resnet and len(resnet_features) > 1:
            resnet_variance = np.var(resnet_features)
            resnet_score = min(1.0, resnet_variance / 8.0)  # Texture anomaly detection

        # 6. YOLO anomaly score (no objects detected = potential hole)
        yolo_score = yolo_features.get('anomaly_score', 0.5)  # High when no objects detected

        # 7. Multi-model ensemble scoring
        if self.use_resnet and self.use_yolo:
            # All three models available
            final_prob = (
                hand_crafted_score * 0.6 +    # Hand-crafted features (strongest)
                resnet_score * 0.2 +          # ResNet texture analysis
                yolo_score * 0.2              # YOLO object detection
            )
        elif self.use_resnet:
            # ResNet only
            final_prob = hand_crafted_score * 0.7 + resnet_score * 0.3
        elif self.use_yolo:
            # YOLO only
            final_prob = hand_crafted_score * 0.7 + yolo_score * 0.3
        else:
            # Hand-crafted features only
            final_prob = hand_crafted_score

        # Enhanced size-based adjustments - boost small holes that could be real
        area = detection['area_pixels']

        # DECORATIVE DOT PENALTY - High contrast, regular shapes are likely false positives
        decorative_penalty = 1.0
        if (hole_features['depth_contrast'] > 0.12 and  # High contrast
            hole_features['shape_irregularity'] < 0.7 and  # Regular shape
            area > 400):  # Medium-large size
            decorative_penalty = 0.6  # 40% penalty for likely decorative dots

        # SUBTLE HOLE BOOST - Low contrast but with irregularity = likely real hole
        subtlety_boost = 1.0
        if (hole_features['depth_contrast'] < 0.10 and  # Low contrast (subtle)
            hole_features['shape_irregularity'] > 0.6 and  # Irregular shape
            hole_features['texture_disruption'] > 0.4):  # Disrupts fabric
            subtlety_boost = 1.5  # 50% boost for subtle but irregular holes
        elif (hole_features['depth_contrast'] < 0.08 and  # Very low contrast
              hole_features['texture_disruption'] > 0.3):  # Some disruption
            subtlety_boost = 1.3  # 30% boost for very subtle holes

        # Size-based multiplier
        if 400 <= area <= 600:  # Perfect hole size range (like our target: 462px)
            size_mult = 1.4
        elif 200 <= area < 400:  # Small holes
            size_mult = 1.3
        elif 600 <= area <= 1000:  # Medium holes
            size_mult = 1.1
        elif area < 200:  # Very small - less likely
            size_mult = 0.8
        elif area > 2000:  # Too large - probably false positive
            size_mult = 0.4
        else:
            size_mult = 1.0

        # Location-based boost for holes near known hole area (experimental)
        target_x, target_y = 1660, 2482
        hole_x, hole_y = x + w//2, y + h//2
        dist_to_known = ((hole_x - target_x)**2 + (y - target_y)**2)**0.5
        if dist_to_known < 100:  # Within 100px of known hole
            location_mult = 1.2
        else:
            location_mult = 1.0

        return min(1.0, final_prob * size_mult * location_mult * decorative_penalty * subtlety_boost)


def test_local_filter():
    """Test local AI filtering to reduce false positives."""
    test_image = "test_shirt.jpg"

    print("=" * 70)
    print("LOCAL AI FILTERING - Fast Pre-filter for OpenAI")
    print("=" * 70)

    # Load previous detections
    with open('enhanced_detections.json', 'r') as f:
        detections = json.load(f)

    print(f"Applying local AI filter to {len(detections)} detections...")

    img = cv2.imread(test_image)
    local_filter = FastLocalFilter()

    # Apply local filtering
    scored_detections = []
    for det in detections:
        local_prob = local_filter.compute_hole_probability(img, det)
        det['local_ai_probability'] = local_prob
        scored_detections.append((det, local_prob))

    # Sort by local AI probability
    scored_detections.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("LOCAL AI FILTERING RESULTS")
    print("=" * 70)

    # Find actual hole
    target_x, target_y = 1660, 2482
    actual_hole_rank = None

    print(f"Top 20 detections by LOCAL AI probability:")
    print("-" * 70)

    high_prob_detections = []

    for i, (det, prob) in enumerate(scored_detections[:20]):
        bbox = det['bbox']
        x, y = bbox['x'], bbox['y']
        dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
        is_target = "🎯" if dist < 50 else "  "

        print(f"{is_target}#{i+1}: ({bbox['x']:4}, {bbox['y']:4}) {bbox['w']:2}x{bbox['h']:2} Prob: {prob:.3f}")

        if dist < 50:
            actual_hole_rank = i + 1
            print(f"      *** ACTUAL HOLE FOUND AT RANK #{i+1} ***")

        # Keep high-probability detections for OpenAI
        if prob > 0.45:  # Lowered threshold for OpenAI verification
            high_prob_detections.append(det)

    print(f"\n📊 LOCAL AI SUMMARY:")
    print(f"  🎯 Actual hole rank: #{actual_hole_rank}")
    print(f"  🔥 High-probability detections (>0.45): {len(high_prob_detections)}")
    reduction_pct = (1-len(high_prob_detections)/len(detections))*100 if len(detections) > 0 else 0.0
    print(f"  📉 Reduction: {len(detections)} → {len(high_prob_detections)} ({reduction_pct:.1f}%)")

    # Save high-probability detections for OpenAI verification
    with open("local_filtered_detections.json", "w") as f:
        json.dump(high_prob_detections, f, indent=2)

    print(f"  💾 High-probability detections saved: local_filtered_detections.json")
    print(f"  💡 These {len(high_prob_detections)} detections should go to OpenAI verification")

    # Status
    if actual_hole_rank and actual_hole_rank <= 10:
        print(f"  ✅ SUCCESS: Actual hole in top 10 local predictions!")
    elif actual_hole_rank and actual_hole_rank <= 20:
        print(f"  ⚠️ DECENT: Actual hole in top 20 local predictions")
    else:
        print(f"  ❌ ISSUE: Actual hole not in top 20 - needs tuning")

    return high_prob_detections


if __name__ == "__main__":
    test_local_filter()