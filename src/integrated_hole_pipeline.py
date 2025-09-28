import cv2
import numpy as np
import json
import time
from typing import List, Dict
from local_ai_filter import FastLocalFilter
from verify_holes_openai import OpenAIHoleVerifier
from verify_holes_enhanced import draw_verified_detections


class IntegratedHolePipeline:
    """
    Complete hole detection pipeline: Local AI Filter → OpenAI Verification
    """

    def __init__(self, openai_api_key: str):
        print("🔧 Initializing Integrated Hole Detection Pipeline...")
        self.local_filter = FastLocalFilter()
        self.openai_verifier = OpenAIHoleVerifier(openai_api_key)
        print("✅ Pipeline ready!")

    def run_complete_pipeline(self, image_path: str,
                            enhanced_detections_path: str = "enhanced_detections.json",
                            local_threshold: float = 0.45,
                            openai_threshold: float = 0.7,
                            max_openai_calls: int = 15) -> List[Dict]:
        """
        Run the complete pipeline: Local AI Filter → OpenAI Verification

        Args:
            image_path: Path to the image
            enhanced_detections_path: Path to enhanced detections JSON
            local_threshold: Threshold for local AI filter
            openai_threshold: Threshold for OpenAI verification
            max_openai_calls: Maximum OpenAI API calls

        Returns:
            List of final verified hole detections
        """

        print("=" * 80)
        print("🏭 INTEGRATED HOLE DETECTION PIPELINE")
        print("=" * 80)
        print(f"📸 Image: {image_path}")
        print(f"🧠 Pipeline: Enhanced Detections → Local AI Filter → OpenAI Verification")

        # Load image and enhanced detections
        image = cv2.imread(image_path)
        with open(enhanced_detections_path, 'r') as f:
            enhanced_detections = json.load(f)

        print(f"\n📊 Starting with {len(enhanced_detections)} enhanced detections")

        # Phase 1: Local AI Filtering
        print(f"\n🔍 [PHASE 1] LOCAL AI FILTERING")
        print("-" * 50)

        local_filtered = self.apply_local_filter(
            image, enhanced_detections, threshold=local_threshold
        )

        print(f"✅ Local AI Filter Results:")
        print(f"   📉 Reduced: {len(enhanced_detections)} → {len(local_filtered)} detections")
        print(f"   💰 Cost savings: ~${(len(enhanced_detections) - len(local_filtered)) * 0.01:.2f}")

        if len(local_filtered) == 0:
            print("⚠️ No detections passed local AI filter")
            return []

        # Phase 2: OpenAI Verification
        print(f"\n🤖 [PHASE 2] OPENAI VERIFICATION")
        print("-" * 50)

        final_detections = self.openai_verifier.verify_detections_batch(
            image, local_filtered,
            max_detections=max_openai_calls,
            min_openai_confidence=openai_threshold
        )

        # Final Results
        print(f"\n{'=' * 80}")
        print(f"🎯 FINAL PIPELINE RESULTS")
        print(f"{'=' * 80}")
        print(f"✨ Final verified holes: {len(final_detections)}")
        reduction_pct = (1 - len(final_detections)/len(enhanced_detections))*100 if len(enhanced_detections) > 0 else 0.0
        print(f"📊 Total reduction: {len(enhanced_detections)} → {len(final_detections)} " +
              f"({reduction_pct:.1f}%)")

        # Check for target hole
        self.check_target_hole(final_detections)

        # Save results
        self.save_results(image_path, final_detections)

        return final_detections

    def apply_local_filter(self, image: np.ndarray, detections: List[Dict],
                          threshold: float = 0.45) -> List[Dict]:
        """Apply local AI filtering with probability scoring."""

        print(f"🔍 Applying local AI filter to {len(detections)} detections...")

        # Score all detections
        scored_detections = []
        for det in detections:
            local_prob = self.local_filter.compute_hole_probability(image, det)
            det['local_ai_probability'] = local_prob
            scored_detections.append((det, local_prob))

        # Sort by probability
        scored_detections.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold
        filtered = []
        target_x, target_y = 1660, 2482
        actual_hole_rank = None

        print(f"\n🔝 Top 10 by local AI probability:")
        for i, (det, prob) in enumerate(scored_detections[:10]):
            bbox = det['bbox']
            x, y = bbox['x'], bbox['y']
            dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
            is_target = "🎯" if dist < 50 else "  "

            print(f"{is_target}#{i+1}: ({x:4}, {y:4}) {bbox['w']:2}x{bbox['h']:2} Prob: {prob:.3f}")

            if dist < 50:
                actual_hole_rank = i + 1
                print(f"      *** ACTUAL HOLE FOUND AT RANK #{i+1} ***")

            if prob >= threshold:
                filtered.append(det)

        print(f"\n📈 Local AI Summary:")
        print(f"   🎯 Actual hole rank: #{actual_hole_rank}")
        print(f"   🔥 Detections above {threshold}: {len(filtered)}")

        return filtered

    def check_target_hole(self, detections: List[Dict]):
        """Check if target hole was found in final results."""
        target_x, target_y = 1660, 2482

        print(f"\n🔍 Checking for target hole at ({target_x}, {target_y})...")
        found = False

        for i, det in enumerate(detections):
            bbox = det['bbox']
            x, y = bbox['x'], bbox['y']
            dist = ((x - target_x)**2 + (y - target_y)**2)**0.5

            if dist < 100:
                openai_result = det.get('openai_verification', {})
                print(f"🎯 FOUND target hole as verified detection #{i+1}!")
                print(f"   📍 Location: ({x}, {y}) - Distance: {dist:.0f}px")
                print(f"   🤖 OpenAI confidence: {openai_result.get('confidence', 0):.2%}")
                print(f"   💬 OpenAI reason: {openai_result.get('reason', 'N/A')}")
                found = True
                break

        if not found:
            print("❌ Target hole not found in final verified detections")

    def save_results(self, image_path: str, detections: List[Dict]):
        """Save final results and visualization."""

        # Save JSON results
        serializable_detections = []
        for det in detections:
            det_copy = {
                'bbox': det['bbox'],
                'confidence': float(det['confidence']),
                'area_pixels': float(det['area_pixels']),
                'final_confidence_score': float(det.get('final_confidence_score', 0)),
                'local_ai_probability': float(det.get('local_ai_probability', 0)),
                'openai_verification': det.get('openai_verification', {})
            }
            serializable_detections.append(det_copy)

        with open("integrated_pipeline_results.json", "w") as f:
            json.dump(serializable_detections, f, indent=2)

        # Save visualization
        output_path = "integrated_pipeline_visualization.jpg"
        draw_verified_detections(image_path, detections, output_path)

        print(f"\n💾 Results saved:")
        print(f"   📄 JSON: integrated_pipeline_results.json")
        print(f"   🖼️ Visualization: {output_path}")


def test_integrated_pipeline(openai_api_key: str):
    """Test the complete integrated pipeline."""

    pipeline = IntegratedHolePipeline(openai_api_key)

    # Run complete pipeline
    final_detections = pipeline.run_complete_pipeline(
        image_path="test_shirt.jpg",
        enhanced_detections_path="enhanced_detections.json",
        local_threshold=0.45,      # Local AI filter threshold
        openai_threshold=0.7,      # OpenAI verification threshold
        max_openai_calls=15        # Cost control
    )

    print(f"\n🏁 Pipeline Complete!")
    print(f"Final result: {len(final_detections)} verified hole(s)")

    return final_detections


if __name__ == "__main__":
    print("🏭 Integrated Hole Detection Pipeline Ready!")
    print("Provide OpenAI API key to run the complete pipeline.")