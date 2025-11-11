"""
Unit tests for AI Model Booster
"""

import torch
import torch.nn as nn
import pytest
import tempfile
import os
from pathlib import Path

from ai_model_booster import (
    AIModelBooster, OptimizationConfig, OptimizationLevel, 
    PrecisionType, AdvancedQuantization, AdvancedPruning
)

class TestModel(nn.Module):
    """Test model for optimization tests"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

class TestAIModelBooster:
    """Test cases for AI Model Booster"""
    
    def setup_method(self):
        """Setup before each test"""
        self.model = TestModel()
        self.config = OptimizationConfig()
        self.booster = AIModelBooster(self.config)
    
    def test_initialization(self):
        """Test booster initialization"""
        assert self.booster is not None
        assert self.booster.config is not None
        assert self.booster.model is None
    
    def test_load_model(self):
        """Test model loading"""
        self.booster.load_model(self.model)
        assert self.booster.model is not None
        assert isinstance(self.booster.model, nn.Module)
    
    def test_light_optimization(self):
        """Test light optimization"""
        self.booster.load_model(self.model)
        optimized_model = self.booster.optimize(OptimizationLevel.LIGHT)
        
        assert optimized_model is not None
        assert self.booster.optimized_model is not None
    
    def test_medium_optimization(self):
        """Test medium optimization"""
        self.booster.load_model(self.model)
        optimized_model = self.booster.optimize(OptimizationLevel.MEDIUM)
        
        assert optimized_model is not None
    
    def test_aggressive_optimization(self):
        """Test aggressive optimization"""
        self.booster.load_model(self.model)
        optimized_model = self.booster.optimize(OptimizationLevel.AGGRESSIVE)
        
        assert optimized_model is not None
    
    def test_benchmark(self):
        """Test benchmarking"""
        self.booster.load_model(self.model)
        self.booster.optimize(OptimizationLevel.MEDIUM)
        
        input_data = torch.randn(1, 100)
        results = self.booster.benchmark(input_data, iterations=10)
        
        assert 'original' in results
        assert 'optimized' in results
        assert 'improvement' in results
        assert 'speedup_factor' in results['improvement']
    
    def test_save_model(self):
        """Test model saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.booster.load_model(self.model)
            self.booster.optimize(OptimizationLevel.MEDIUM)
            
            output_path = os.path.join(temp_dir, 'test_model.pth')
            self.booster.save_optimized_model(output_path)
            
            assert os.path.exists(output_path)
            assert os.path.exists(output_path.replace('.pth', '_report.json'))
    
    def test_optimization_report(self):
        """Test optimization report generation"""
        self.booster.load_model(self.model)
        self.booster.optimize(OptimizationLevel.MEDIUM)
        
        report = self.booster.get_optimization_report()
        assert isinstance(report, dict)
        assert 'benchmark' in report

class TestAdvancedQuantization:
    """Test cases for AdvancedQuantization"""
    
    def setup_method(self):
        self.quantizer = AdvancedQuantization()
        self.model = TestModel()
    
    def test_dynamic_quantization(self):
        """Test dynamic quantization"""
        quantized_model = self.quantizer.apply_dynamic_quantization(self.model)
        assert quantized_model is not None
    
    def test_calibration_preparation(self):
        """Test calibration data preparation"""
        dataloader = [torch.randn(1, 100) for _ in range(5)]
        self.quantizer.prepare_calibration(self.model, dataloader, num_samples=5)
        assert self.quantizer.calibration_data is not None
        assert len(self.quantizer.calibration_data) == 5

class TestAdvancedPruning:
    """Test cases for AdvancedPruning"""
    
    def setup_method(self):
        self.pruner = AdvancedPruning()
        self.model = TestModel()
    
    def test_structured_pruning(self):
        """Test structured pruning"""
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_model = self.pruner.apply_structured_pruning(self.model, 0.2)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        assert pruned_model is not None
        # Pruning should reduce some parameters (though not exactly 20% due to structure)
        assert pruned_params <= original_params
    
    def test_magnitude_pruning(self):
        """Test magnitude pruning"""
        pruned_model = self.pruner.apply_magnitude_pruning(self.model, 0.01)
        assert pruned_model is not None

def test_optimization_pipeline():
    """Test the complete optimization pipeline"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and save test model
        model = TestModel()
        model_path = os.path.join(temp_dir, 'test_model.pth')
        torch.save(model.state_dict(), model_path)
        
        output_path = os.path.join(temp_dir, 'optimized_model.pth')
        
        # Run pipeline
        from ai_model_booster import optimize_model_pipeline
        booster = optimize_model_pipeline(
            model_path=model_path,
            output_path=output_path,
            optimization_level=OptimizationLevel.MEDIUM
        )
        
        assert booster is not None
        assert os.path.exists(output_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
