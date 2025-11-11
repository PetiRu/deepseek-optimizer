"""
Basic usage examples for AI Model Booster
"""

import torch
import torch.nn as nn
from ai_model_booster import AIModelBooster, OptimizationConfig, OptimizationLevel, PrecisionType

# Example 1: Simple model optimization
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def example_basic_optimization():
    """Basic optimization example"""
    print("🚀 Example 1: Basic Model Optimization")
    
    # Create model
    model = SimpleModel()
    
    # Create optimizer with default config
    booster = AIModelBooster()
    booster.load_model(model)
    
    # Apply medium optimization
    optimized_model = booster.optimize(OptimizationLevel.MEDIUM)
    
    # Benchmark
    input_data = torch.randn(1, 784)
    results = booster.benchmark(input_data)
    
    print("📊 Optimization Results:")
    print(f"Speed Improvement: {results['improvement']['speedup_factor']:.2f}x")
    print(f"Memory Reduction: {results['improvement']['memory_reduction_percent']:.2f}%")
    
    return booster

def example_custom_config():
    """Example with custom configuration"""
    print("\n🚀 Example 2: Custom Configuration")
    
    # Create custom configuration
    config = OptimizationConfig(
        precision=PrecisionType.FP16,
        level=OptimizationLevel.AGGRESSIVE,
        pruning_ratio=0.3,
        use_quantization=True,
        use_pruning=True,
        use_compilation=True,
        target_device="cuda"
    )
    
    model = SimpleModel()
    booster = AIModelBooster(config)
    booster.load_model(model)
    
    optimized_model = booster.optimize()
    results = booster.benchmark(torch.randn(1, 784))
    
    print("📊 Custom Configuration Results:")
    print(f"Speed Improvement: {results['improvement']['speedup_factor']:.2f}x")
    print(f"Model Size Reduction: {results['improvement']['size_reduction_percent']:.2f}%")
    
    return booster

def example_pipeline_usage():
    """Example using the optimization pipeline"""
    print("\n🚀 Example 3: Optimization Pipeline")
    
    # Save model first
    model = SimpleModel()
    torch.save(model.state_dict(), 'simple_model.pth')
    
    # Use pipeline
    from ai_model_booster import optimize_model_pipeline
    
    booster = optimize_model_pipeline(
        model_path='simple_model.pth',
        output_path='optimized_simple_model.pth',
        optimization_level=OptimizationLevel.AGGRESSIVE
    )
    
    report = booster.get_optimization_report()
    print("✅ Pipeline completed successfully!")
    print(f"Benchmark results: {report['benchmark']['improvement']}")
    
    return booster

if __name__ == "__main__":
    # Run all examples
    example_basic_optimization()
    example_custom_config()
    example_pipeline_usage()
