"""
Hardware Configuration for RTX 5090 Deployment
Optimizes model loading and GPU utilization for high-end hardware.
"""

import torch
import psutil
import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """Configuration for RTX 5090 deployment."""
    num_gpus: int
    total_vram_gb: float
    cpu_cores: int
    ram_gb: float
    storage_speed_gbps: float

    # Model configuration
    use_large_models: bool = True
    use_fp16: bool = True
    use_flash_attention: bool = True
    batch_size: int = 8
    max_sequence_length: int = 512


class RTX5090Optimizer:
    """Optimize AI models for RTX 5090 hardware."""

    def __init__(self):
        self.config = self.detect_hardware()
        self.setup_torch_optimizations()

    def detect_hardware(self) -> HardwareConfig:
        """Detect and analyze available hardware."""
        print("🔍 Analyzing hardware configuration...")

        # GPU detection
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        total_vram_gb = 0.0

        if num_gpus > 0:
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                total_vram_gb += vram_gb
                print(f"   GPU {i}: {props.name} ({vram_gb:.1f}GB VRAM)")

        # CPU and RAM
        cpu_cores = psutil.cpu_count(logical=False)
        ram_gb = psutil.virtual_memory().total / (1024**3)

        print(f"   CPU: {cpu_cores} cores")
        print(f"   RAM: {ram_gb:.1f}GB")
        print(f"   Total VRAM: {total_vram_gb:.1f}GB")

        # Estimate storage speed (placeholder - would need benchmarking)
        storage_speed_gbps = 6.4  # From your specs

        config = HardwareConfig(
            num_gpus=num_gpus,
            total_vram_gb=total_vram_gb,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            storage_speed_gbps=storage_speed_gbps
        )

        # Optimize settings based on hardware
        self.optimize_config(config)

        return config

    def optimize_config(self, config: HardwareConfig):
        """Optimize configuration based on detected hardware."""

        # Enable advanced features for RTX 5090
        if config.total_vram_gb >= 60:  # 2x RTX 5090
            print("🚀 RTX 5090 configuration detected - enabling advanced features")
            config.use_large_models = True
            config.use_fp16 = True
            config.use_flash_attention = True
            config.batch_size = 16  # Larger batches for better GPU utilization
            config.max_sequence_length = 1024

        elif config.total_vram_gb >= 30:  # 1x RTX 5090
            print("🔥 High-end GPU detected - enabling optimized features")
            config.use_large_models = True
            config.use_fp16 = True
            config.batch_size = 12
            config.max_sequence_length = 768

        else:
            print("⚡ Standard configuration")
            config.use_large_models = False
            config.batch_size = 4
            config.max_sequence_length = 512

    def setup_torch_optimizations(self):
        """Setup PyTorch optimizations for RTX 5090."""
        if torch.cuda.is_available():
            # Enable optimizations for RTX 5090
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Enable TensorFloat-32 for RTX 5090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable optimized attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("✅ Flash attention enabled")
            except:
                pass

            print("🔧 PyTorch optimizations enabled for RTX 5090")

    def get_optimal_device_map(self, num_models: int) -> Dict[str, str]:
        """Get optimal device mapping for multiple models."""
        device_map = {}

        if self.config.num_gpus >= 2:
            # Distribute models across GPUs
            device_assignments = [
                "cuda:0",  # YOLO on primary GPU
                "cuda:1",  # Vision backbone on secondary GPU
                "cuda:0",  # CLIP on primary GPU (shared)
                "cuda:1",  # VLM on secondary GPU (shared)
                "cuda:0"   # Depth on primary GPU
            ]

            models = ["yolo", "vision", "clip", "vlm", "depth"]
            for i, model in enumerate(models[:num_models]):
                device_map[model] = device_assignments[i]

        else:
            # Single GPU - all models on same device
            for model in ["yolo", "vision", "clip", "vlm", "depth"]:
                device_map[model] = "cuda:0" if self.config.num_gpus > 0 else "cpu"

        return device_map

    def get_model_recommendations(self) -> Dict[str, str]:
        """Get recommended models based on hardware capabilities."""
        recommendations = {}

        if self.config.total_vram_gb >= 60:
            # RTX 5090 x2 - Use largest models
            recommendations = {
                "yolo": "yolo11x.pt",  # Largest YOLO
                "vision": "google/efficientnet-b7",  # Large vision model
                "clip": "openai/clip-vit-large-patch14",  # Large CLIP
                "vlm": "Salesforce/blip2-opt-6.7b",  # Larger VLM
                "depth": "Intel/dpt-large"  # Large depth model
            }
        elif self.config.total_vram_gb >= 30:
            # RTX 5090 x1 - Balanced large models
            recommendations = {
                "yolo": "yolo11l.pt",  # Large YOLO
                "vision": "microsoft/resnet-152",  # Large ResNet
                "clip": "openai/clip-vit-large-patch14",
                "vlm": "Salesforce/blip2-opt-2.7b",
                "depth": "Intel/dpt-hybrid-midas"
            }
        else:
            # Fallback to current models
            recommendations = {
                "yolo": "yolo11n.pt",
                "vision": "microsoft/resnet-18",
                "clip": "openai/clip-vit-base-patch32",
                "vlm": None,  # Skip VLM on limited hardware
                "depth": None
            }

        return recommendations

    def print_optimization_summary(self):
        """Print hardware optimization summary."""
        print("\n" + "="*60)
        print("🚀 RTX 5090 OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"🔥 GPUs: {self.config.num_gpus} ({self.config.total_vram_gb:.1f}GB total VRAM)")
        print(f"🧠 CPU: {self.config.cpu_cores} cores")
        print(f"💾 RAM: {self.config.ram_gb:.1f}GB")
        print(f"💿 Storage: {self.config.storage_speed_gbps:.1f}GB/s")
        print(f"⚡ Large models: {'✅' if self.config.use_large_models else '❌'}")
        print(f"🔥 FP16 precision: {'✅' if self.config.use_fp16 else '❌'}")
        print(f"⚡ Flash attention: {'✅' if self.config.use_flash_attention else '❌'}")
        print(f"📦 Batch size: {self.config.batch_size}")

        recommendations = self.get_model_recommendations()
        print(f"\n📋 Recommended Models:")
        for model_type, model_name in recommendations.items():
            if model_name:
                print(f"   {model_type}: {model_name}")

        device_map = self.get_optimal_device_map(5)
        print(f"\n🎯 Device Mapping:")
        for model, device in device_map.items():
            print(f"   {model}: {device}")


def save_hardware_config(optimizer: RTX5090Optimizer, filepath: str):
    """Save hardware configuration to file."""
    config_dict = {
        "hardware": {
            "num_gpus": optimizer.config.num_gpus,
            "total_vram_gb": optimizer.config.total_vram_gb,
            "cpu_cores": optimizer.config.cpu_cores,
            "ram_gb": optimizer.config.ram_gb,
            "storage_speed_gbps": optimizer.config.storage_speed_gbps
        },
        "optimization": {
            "use_large_models": optimizer.config.use_large_models,
            "use_fp16": optimizer.config.use_fp16,
            "use_flash_attention": optimizer.config.use_flash_attention,
            "batch_size": optimizer.config.batch_size,
            "max_sequence_length": optimizer.config.max_sequence_length
        },
        "models": optimizer.get_model_recommendations(),
        "device_mapping": optimizer.get_optimal_device_map(5)
    }

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"💾 Hardware configuration saved to: {filepath}")


if __name__ == "__main__":
    optimizer = RTX5090Optimizer()
    optimizer.print_optimization_summary()
    save_hardware_config(optimizer, "../data/rtx5090_config.json")