"""
Performance benchmarking for AI Model Booster
"""

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ai_model_booster import AIModelBooster, OptimizationConfig, OptimizationLevel

class BenchmarkModel(nn.Module):
    """Benchmark model for performance testing"""
    
    def __init__(self, size="medium"):
        super().__init__()
        
        if size == "small":
            layers = [
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            ]
        elif size == "medium":
            layers = [
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            ]
        else:  # large
            layers = [
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            ]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def benchmark_optimization_levels():
    """Benchmark different optimization levels"""
    print("📊 Benchmarking Optimization Levels")
    
    model_sizes = ["small", "medium", "large"]
    optimization_levels = [OptimizationLevel.LIGHT, OptimizationLevel.MEDIUM, OptimizationLevel.AGGRESSIVE]
    
    results = {}
    
    for size in model_sizes:
        print(f"\n🔍 Testing {size} model...")
        results[size] = {}
        
        model = BenchmarkModel(size)
        input_shape = (1, 1024) if size == "small" else (1, 1024) if size == "medium" else (1, 2048)
        input_data = torch.randn(input_shape)
        
        for level in optimization_levels:
            print(f"  Optimizing with {level.value} level...")
            
            booster = AIModelBooster()
            booster.load_model(model)
            booster.optimize(level)
            
            benchmark_results = booster.benchmark(input_data, iterations=50)
            results[size][level.value] = benchmark_results['improvement']
            
            print(f"    Speedup: {benchmark_results['improvement']['speedup_factor']:.2f}x")
    
    return results

def benchmark_memory_usage():
    """Benchmark memory usage across different optimizations"""
    print("\n💾 Benchmarking Memory Usage")
    
    model = BenchmarkModel("large")
    input_data = torch.randn(1, 2048)
    
    memory_results = {}
    
    # Test different optimization configurations
    configs = {
        "baseline": OptimizationConfig(use_quantization=False, use_pruning=False, use_compilation=False),
        "quantization_only": OptimizationConfig(use_quantization=True, use_pruning=False, use_compilation=False),
        "pruning_only": OptimizationConfig(use_quantization=False, use_pruning=True, use_compilation=False),
        "compilation_only": OptimizationConfig(use_quantization=False, use_pruning=False, use_compilation=True),
        "full_optimization": OptimizationConfig(use_quantization=True, use_pruning=True, use_compilation=True)
    }
    
    for config_name, config in configs.items():
        print(f"Testing {config_name}...")
        
        booster = AIModelBooster(config)
        booster.load_model(model)
        booster.optimize(OptimizationLevel.MEDIUM)
        
        results = booster.benchmark(input_data, iterations=30)
        memory_results[config_name] = {
            'memory_mb': results['optimized']['memory_usage_mb'],
            'inference_time_ms': results['optimized']['inference_time_ms']
        }
    
    return memory_results

def benchmark_precision_types():
    """Benchmark different precision types"""
    print("\n🎯 Benchmarking Precision Types")
    
    model = BenchmarkModel("medium")
    input_data = torch.randn(1, 1024)
    
    precision_results = {}
    
    from ai_model_booster import PrecisionType
    
    precision_configs = {
        "FP32": OptimizationConfig(precision=PrecisionType.FP32),
        "FP16": OptimizationConfig(precision=PrecisionType.FP16),
        "INT8": OptimizationConfig(precision=PrecisionType.INT8)
    }
    
    for precision_name, config in precision_configs.items():
        print(f"Testing {precision_name}...")
        
        booster = AIModelBooster(config)
        booster.load_model(model)
        booster.optimize(OptimizationLevel.MEDIUM)
        
        results = booster.benchmark(input_data, iterations=50)
        precision_results[precision_name] = {
            'inference_time_ms': results['optimized']['inference_time_ms'],
            'memory_usage_mb': results['optimized']['memory_usage_mb'],
            'model_size_mb': results['optimized']['model_size_mb']
        }
        
        print(f"  Time: {results['optimized']['inference_time_ms']:.3f}ms")
        print(f"  Memory: {results['optimized']['memory_usage_mb']:.1f}MB")
    
    return precision_results

def create_visualizations(results):
    """Create visualization charts from benchmark results"""
    print("\n📈 Creating Visualizations")
    
    # Optimization levels comparison
    if 'small' in results:
        levels = list(results['small'].keys())
        speedups_small = [results['small'][level]['speedup_factor'] for level in levels]
        speedups_medium = [results['medium'][level]['speedup_factor'] for level in levels]
        speedups_large = [results['large'][level]['speedup_factor'] for level in levels]
        
        x = np.arange(len(levels))
        width = 0.25
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(x - width, speedups_small, width, label='Small Model')
        plt.bar(x, speedups_medium, width, label='Medium Model')
        plt.bar(x + width, speedups_large, width, label='Large Model')
        plt.xlabel('Optimization Level')
        plt.ylabel('Speedup Factor (x)')
        plt.title('Optimization Level Comparison')
        plt.xticks(x, levels)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Save visualizations
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_comprehensive_benchmark():
    """Run all benchmarks and generate report"""
    print("🚀 Starting Comprehensive Benchmark")
    print("=" * 50)
    
    all_results = {}
    
    # Run all benchmarks
    all_results['optimization_levels'] = benchmark_optimization_levels()
    all_results['memory_usage'] = benchmark_memory_usage()
    all_results['precision_types'] = benchmark_precision_types()
    
    # Save results
    with open('comprehensive_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create visualizations
    create_visualizations(all_results['optimization_levels'])
    
    # Print summary
    print("\n🎉 Benchmark Summary")
    print("=" * 50)
    
    best_speedup = 0
    best_config = ""
    
    for size, levels in all_results['optimization_levels'].items():
        for level, results in levels.items():
            speedup = results['speedup_factor']
            if speedup > best_speedup:
                best_speedup = speedup
                best_config = f"{size} model with {level} optimization"
    
    print(f"🏆 Best Performance: {best_speedup:.2f}x speedup")
    print(f"🏆 Best Configuration: {best_config}")
    
    # Memory results
    if 'memory_usage' in all_results:
        best_memory = min([r['memory_mb'] for r in all_results['memory_usage'].values()])
        print(f"💾 Best Memory Usage: {best_memory:.1f} MB")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    print("\n✅ Benchmark completed! Check 'comprehensive_benchmark_results.json' for detailed results.")
