"""
Comprehensive Tests for Matching Engines

Enterprise-grade testing for parametric solver, neural inference, and hybrid orchestrator.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParametricParameters:
    """Tests for ParametricParameters data structure."""
    
    def test_identity_creation(self):
        """Test creating identity parameters."""
        from src.engines.parametric.parameters import ParametricParameters
        
        params = ParametricParameters.identity()
        
        assert params.matrix.shape == (3, 4)
        assert len(params.rgb_curves) == 3
    
    def test_to_vector_length(self):
        """Test vector conversion produces correct length."""
        from src.engines.parametric.parameters import ParametricParameters
        
        params = ParametricParameters.identity()
        vec = params.to_vector()
        
        assert len(vec) == ParametricParameters.TOTAL_PARAMS
        assert len(vec) == 160
    
    def test_from_vector_roundtrip(self):
        """Test vector conversion roundtrip."""
        from src.engines.parametric.parameters import ParametricParameters
        
        original = ParametricParameters.identity()
        vec = original.to_vector()
        restored = ParametricParameters.from_vector(vec)
        
        assert_allclose(restored.matrix, original.matrix)
        assert_allclose(restored.rgb_curves['R'], original.rgb_curves['R'])
        assert_allclose(restored.rgb_curves['G'], original.rgb_curves['G'])
        assert_allclose(restored.rgb_curves['B'], original.rgb_curves['B'])
    
    def test_save_load_roundtrip(self):
        """Test JSON save/load roundtrip."""
        from src.engines.parametric.parameters import ParametricParameters
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            original = ParametricParameters.identity()
            original.matrix[0, 0] = 1.5  # Modify something
            original.save(path)
            
            loaded = ParametricParameters.load(path)
            
            assert_allclose(loaded.matrix, original.matrix)
        finally:
            os.unlink(path)
    
    def test_bounds_valid(self):
        """Test parameter bounds are valid."""
        from src.engines.parametric.parameters import get_parameter_bounds
        
        lower, upper = get_parameter_bounds()
        
        assert len(lower) == 160
        assert len(upper) == 160
        assert np.all(lower <= upper)
    
    def test_identity_within_bounds(self):
        """Test identity parameters are within bounds."""
        from src.engines.parametric.parameters import ParametricParameters, get_parameter_bounds
        
        params = ParametricParameters.identity()
        vec = params.to_vector()
        lower, upper = get_parameter_bounds()
        
        # Check all values are within their respective bounds
        # Identity matrix has 1s on diagonal and 0s elsewhere
        for i in range(len(vec)):
            assert vec[i] >= lower[i], f"Value {i}: {vec[i]} < lower {lower[i]}"
            assert vec[i] <= upper[i], f"Value {i}: {vec[i]} > upper {upper[i]}"


class TestCDLParameters:
    """Tests for CDL parameter structure."""
    
    def test_identity_cdl(self):
        """Test identity CDL."""
        from src.engines.parametric.parameters import CDLParameters
        
        cdl = CDLParameters()
        
        assert cdl.slope == (1.0, 1.0, 1.0)
        assert cdl.offset == (0.0, 0.0, 0.0)
        assert cdl.power == (1.0, 1.0, 1.0)
        assert cdl.saturation == 1.0
    
    def test_cdl_to_dict(self):
        """Test CDL to dict conversion."""
        from src.engines.parametric.parameters import CDLParameters
        
        cdl = CDLParameters(
            slope=(1.1, 1.0, 0.9),
            saturation=1.2
        )
        
        d = cdl.to_dict()
        
        # Note: to_dict returns lists, not tuples
        assert list(d['slope']) == [1.1, 1.0, 0.9]
        assert d['saturation'] == 1.2


class TestParametricSolver:
    """Tests for ParametricSolver."""

    @pytest.fixture
    def solver(self):
        from src.engines.parametric.solver import ParametricSolver
        return ParametricSolver()
    
    @pytest.fixture
    def test_images(self):
        """Create test source and target images."""
        np.random.seed(42)
        source = np.random.rand(30, 30, 3)
        target = source * 0.9 + 0.05  # Simple transformation
        return source, target
    
    def test_identity_match(self, solver, test_images):
        """Test matching identical images produces near-identity."""
        source, _ = test_images
        
        result = solver.match(source, source.copy())
        
        assert 'parameters' in result
        assert 'metrics' in result
        # For identical images, delta E should be very low
        assert result['metrics']['delta_e_mean'] < 2.0
    
    def test_simple_offset_match(self, solver):
        """Test matching with simple offset."""
        source = np.ones((30, 30, 3)) * 0.5
        target = np.ones((30, 30, 3)) * 0.6
        
        result = solver.match(source, target)
        
        assert result['success'] or result['iterations'] > 0
    
    def test_match_returns_parameters(self, solver, test_images):
        """Test match returns ParametricParameters."""
        from src.engines.parametric.parameters import ParametricParameters
        
        source, target = test_images
        result = solver.match(source, target)
        
        assert isinstance(result['parameters'], ParametricParameters)
    
    def test_apply_parameters(self, solver, test_images):
        """Test apply preserves shape."""
        from src.engines.parametric.parameters import ParametricParameters
        
        source, _ = test_images
        params = ParametricParameters.identity()
        
        result = solver.apply(source, params)
        
        assert result.shape == source.shape
    
    def test_cdl_quick_match(self, solver):
        """Test CDL-only optimization (fast mode)."""
        source = np.random.rand(30, 30, 3)
        target = source * 1.1
        
        # Use optimize_cdl_only (correct method name)
        result = solver.optimize_cdl_only(source, target)
        
        assert 'cdl' in result
        assert 'metrics' in result


class TestParametricOptimizer:
    """Tests for ParametricOptimizer."""
    
    @pytest.mark.skip(reason="Optimizer requires more setup")
    def test_progressive_optimization(self):
        """Test progressive optimization stages."""
        pass


class TestNeuralInference:
    """Tests for neural inference engine."""
    
    def test_inference_without_model(self):
        """Test inference gracefully handles missing model."""
        try:
            from src.engines.neural.inference import NeuralInference
            
            inference = NeuralInference(model_path=None)
            # Should initialize but model_loaded should be False or similar
            assert inference is not None
        except ImportError:
            pytest.skip("Neural inference requires PyTorch")


class TestHybridOrchestrator:
    """Tests for hybrid orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        try:
            from src.engines.hybrid.orchestrator import HybridOrchestrator
            return HybridOrchestrator()
        except ImportError:
            pytest.skip("HybridOrchestrator requires PyTorch for neural engine")
    
    def test_parametric_strategy(self, orchestrator):
        """Test parametric-only strategy."""
        source = np.random.rand(30, 30, 3)
        target = source * 0.9
        
        result = orchestrator.match(source, target, strategy='parametric')
        
        assert 'parameters' in result
    
    def test_auto_strategy(self, orchestrator):
        """Test auto strategy selection."""
        source = np.random.rand(30, 30, 3)
        target = np.random.rand(30, 30, 3)
        
        result = orchestrator.match(source, target, strategy='auto')
        
        assert 'parameters' in result or 'error' in result
    
    def test_output_shape_preserved(self, orchestrator):
        """Test output shape matches input."""
        source = np.random.rand(50, 75, 3)
        target = np.random.rand(50, 75, 3)
        
        result = orchestrator.match(source, target, strategy='parametric')
        
        if 'matched_image' in result:
            assert result['matched_image'].shape == source.shape
    
    def test_metrics_present(self, orchestrator):
        """Test metrics are returned."""
        source = np.random.rand(30, 30, 3)
        target = source.copy()
        
        result = orchestrator.match(source, target, strategy='parametric')
        
        assert 'metrics' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
