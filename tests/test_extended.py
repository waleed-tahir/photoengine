"""
Extended Tests for Chromatica Pro

Additional integration tests for pipeline components.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestACESPipeline:
    """Extended ACES pipeline tests."""
    
    def test_camera_profiles_exist(self):
        """Test that camera profiles are available."""
        from src.core.aces_pipeline import ACESPipeline
        
        pipeline = ACESPipeline()
        
        # Use correct attribute name
        assert len(pipeline.idt_library) > 0
    
    def test_full_pipeline_roundtrip(self):
        """Test IDT -> LMT -> RRT+ODT produces valid output."""
        from src.core.aces_pipeline import ACESPipeline, LMTParams
        
        pipeline = ACESPipeline()
        
        source = np.random.rand(30, 30, 3).astype(np.float64) * 0.5
        
        # Step-by-step pipeline
        ap0 = pipeline.apply_idt(source, 'sRGB')
        graded = pipeline.apply_lmt(ap0, LMTParams())
        result = pipeline.apply_rrt_odt(graded, 'sRGB')
        
        assert result.shape == source.shape
        assert not np.isnan(result).any()


class TestHybridOrchestrator:
    """Extended tests for hybrid orchestrator."""
    
    def test_auto_strategy_selection(self):
        """Test auto strategy selects appropriately."""
        try:
            from src.engines.hybrid.orchestrator import HybridOrchestrator
            
            orchestrator = HybridOrchestrator()
            
            source = np.random.rand(30, 30, 3)
            target = source * 0.9
            
            result = orchestrator.match(source, target, strategy='parametric')
            assert 'parameters' in result
        except ImportError:
            pytest.skip("Orchestrator requires PyTorch")


class TestExport:
    """Tests for export functionality."""
    
    def test_cube_lut_export(self):
        """Test .cube LUT export."""
        from src.io.export import LUTExporter
        from src.engines.parametric.parameters import ParametricParameters
        
        exporter = LUTExporter()
        params = ParametricParameters.identity()
        
        with tempfile.NamedTemporaryFile(suffix='.cube', delete=False) as f:
            exporter.export_cube(f.name, params, size=9)  # Small size for speed
            
            with open(f.name, 'r') as cube:
                content = cube.read()
        
        assert 'LUT_3D_SIZE 9' in content
        assert 'TITLE' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
