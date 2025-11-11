"""
Optimization configuration presets for different use cases
"""

from dataclasses import dataclass
from enum import Enum
from ai_model_booster import OptimizationConfig, OptimizationLevel, PrecisionType

@dataclass
class DeepSeekOptimizationPresets:
    """Optimization presets specifically for DeepSeek models"""
    
    @staticmethod
    def get_coder_config():
        """Configuration for DeepSeek-Coder models"""
        return OptimizationConfig(
            precision=PrecisionType.FP16,
            level=OptimizationLevel.AGGRESSIVE,
            pruning_ratio=0.1,
            use_quantization=True,
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=True,
            use_memory_opt=True,
            target_device="cuda"
        )
    
    @staticmethod
    def get_v2_config():
        """Configuration for DeepSeek-V2 models"""
        return OptimizationConfig(
            precision=PrecisionType.INT8,
            level=OptimizationLevel.AGGRESSIVE,
            pruning_ratio=0.05,  # Less pruning for MoE models
            use_quantization=True,
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=True,
            use_memory_opt=True,
            target_device="cuda"
        )
    
    @staticmethod
    def get_math_config():
        """Configuration for DeepSeek-Math models"""
        return OptimizationConfig(
            precision=PrecisionType.FP16,
            level=OptimizationLevel.MEDIUM,  # More conservative for math precision
            pruning_ratio=0.1,
            use_quantization=True,
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=True,
            use_memory_opt=True,
            target_device="cuda"
        )

@dataclass
class HardwareSpecificPresets:
    """Hardware-specific optimization presets"""
    
    @staticmethod
    def get_nvidia_config():
        """Optimization for NVIDIA GPUs"""
        return OptimizationConfig(
            precision=PrecisionType.FP16,
            level=OptimizationLevel.AGGRESSIVE,
            pruning_ratio=0.2,
            use_quantization=True,
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=True,
            use_memory_opt=True,
            target_device="cuda"
        )
    
    @staticmethod
    def get_amd_config():
        """Optimization for AMD GPUs"""
        return OptimizationConfig(
            precision=PrecisionType.FP32,  # Better compatibility
            level=OptimizationLevel.MEDIUM,
            pruning_ratio=0.15,
            use_quantization=False,  # May have issues on AMD
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=False,  # May not be supported
            use_memory_opt=True,
            target_device="cpu"  # Fallback to CPU optimizations
        )
    
    @staticmethod
    def get_cpu_config():
        """Optimization for CPU inference"""
        return OptimizationConfig(
            precision=PrecisionType.INT8,
            level=OptimizationLevel.AGGRESSIVE,
            pruning_ratio=0.3,
            use_quantization=True,
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=False,
            use_memory_opt=True,
            target_device="cpu"
        )

@dataclass
class UseCasePresets:
    """Optimization presets for specific use cases"""
    
    @staticmethod
    def get_low_latency_config():
        """Configuration for low-latency applications"""
        return OptimizationConfig(
            precision=PrecisionType.FP16,
            level=OptimizationLevel.AGGRESSIVE,
            pruning_ratio=0.1,
            use_quantization=True,
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=True,
            use_memory_opt=True,
            target_device="cuda"
        )
    
    @staticmethod
    def get_memory_constrained_config():
        """Configuration for memory-constrained environments"""
        return OptimizationConfig(
            precision=PrecisionType.INT8,
            level=OptimizationLevel.AGGRESSIVE,
            pruning_ratio=0.4,  # Heavy pruning
            use_quantization=True,
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=True,
            use_memory_opt=True,
            target_device="cpu"
        )
    
    @staticmethod
    def get_high_accuracy_config():
        """Configuration for high-accuracy requirements"""
        return OptimizationConfig(
            precision=PrecisionType.FP32,
            level=OptimizationLevel.LIGHT,  # Minimal changes to preserve accuracy
            pruning_ratio=0.05,  # Very light pruning
            use_quantization=False,  # No quantization for max accuracy
            use_pruning=True,
            use_compilation=True,
            use_kernel_fusion=True,
            use_memory_opt=True,
            target_device="cuda"
        )

# Export presets
PRESETS = {
    # DeepSeek specific
    "deepseek-coder": DeepSeekOptimizationPresets.get_coder_config(),
    "deepseek-v2": DeepSeekOptimizationPresets.get_v2_config(),
    "deepseek-math": DeepSeekOptimizationPresets.get_math_config(),
    
    # Hardware specific
    "nvidia": HardwareSpecificPresets.get_nvidia_config(),
    "amd": HardwareSpecificPresets.get_amd_config(),
    "cpu": HardwareSpecificPresets.get_cpu_config(),
    
    # Use case specific
    "low-latency": UseCasePresets.get_low_latency_config(),
    "memory-constrained": UseCasePresets.get_memory_constrained_config(),
    "high-accuracy": UseCasePresets.get_high_accuracy_config(),
}

def get_preset_config(preset_name: str) -> OptimizationConfig:
    """Get optimization configuration by preset name"""
    if preset_name not in PRESETS:
        available_presets = list(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    return PRESETS[preset_name]

def list_available_presets() -> list:
    """List all available optimization presets"""
    return list(PRESETS.keys())
