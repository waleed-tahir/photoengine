"""
Import Validation Tests

Tests to ensure all critical imports work correctly and will be bundled by PyInstaller.
These tests should be run before building to catch import issues early.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreImports:
    """Test that all core scientific libraries can be imported."""
    
    def test_numpy_import(self):
        """Test numpy and its submodules."""
        import numpy
        import numpy.core
        import numpy.linalg
        import numpy.random
        
        # Verify numpy works
        arr = numpy.array([1, 2, 3])
        assert arr.sum() == 6
    
    def test_numpy_typing(self):
        """Test numpy typing extension."""
        import numpy.typing as npt
        import numpy as np
        
        # Verify types work
        x: npt.NDArray[np.float64] = np.zeros(5)
        assert len(x) == 5
    
    def test_scipy_import(self):
        """Test scipy and key submodules."""
        try:
            import scipy
            from scipy import optimize
            from scipy import interpolate
            from scipy import ndimage
            assert True
        except ImportError:
            pytest.skip("scipy not installed")
    
    def test_cv2_import(self):
        """Test OpenCV import."""
        try:
            import cv2
            assert hasattr(cv2, 'imread')
        except ImportError:
            pytest.skip("opencv-python not installed")
    
    def test_pil_import(self):
        """Test PIL/Pillow import."""
        try:
            from PIL import Image
            assert hasattr(Image, 'open')
        except ImportError:
            pytest.skip("Pillow not installed")


class TestApplicationModuleImports:
    """Test that all application modules can be imported."""
    
    def test_core_color_spaces(self):
        """Test core.color_spaces module."""
        from src.core.color_spaces import ColorSpaceConverter
        
        converter = ColorSpaceConverter()
        assert converter is not None
    
    def test_core_color_science(self):
        """Test core.color_science module."""
        from src.core.color_science import ColorScience
        
        assert hasattr(ColorScience, 'delta_e_2000')
    
    def test_core_transforms(self):
        """Test core.transforms module."""
        from src.core.transforms import ColorTransforms, LUT3D
        
        assert hasattr(ColorTransforms, 'apply_cdl')
        assert LUT3D is not None
    
    def test_core_aces_pipeline(self):
        """Test core.aces_pipeline module."""
        from src.core.aces_pipeline import ACESPipeline, LMTParams
        
        pipeline = ACESPipeline()
        assert len(pipeline.idt_library) > 0
    
    def test_engines_parametric(self):
        """Test engines.parametric module."""
        from src.engines.parametric.parameters import ParametricParameters, CDLParameters
        from src.engines.parametric.solver import ParametricSolver
        
        params = ParametricParameters.identity()
        assert params.matrix.shape == (3, 4)
    
    def test_io_formats(self):
        """Test io.formats module."""
        from src.io.formats import ImageIO
        
        assert hasattr(ImageIO, 'read')
        assert hasattr(ImageIO, 'write')
    
    def test_io_export(self):
        """Test io.export module."""
        from src.io.export import LUTExporter, CDLExporter
        
        lut_exp = LUTExporter()
        cdl_exp = CDLExporter()
        assert lut_exp is not None
        assert cdl_exp is not None
    
    def test_analysis_validation(self):
        """Test analysis.validation module."""
        from src.analysis.validation import ColorCheckerValidator, COLORCHECKER_CLASSIC
        
        assert len(COLORCHECKER_CLASSIC) == 24
    
    def test_analysis_quality(self):
        """Test analysis.quality module."""
        from src.analysis.quality import QualityAnalyzer
        
        analyzer = QualityAnalyzer()
        assert analyzer is not None
    
    def test_analysis_scopes(self):
        """Test analysis.scopes module."""
        from src.analysis.scopes import WaveformScope, VectorscopeRenderer, HistogramRenderer
        
        wf = WaveformScope(360, 240)
        vs = VectorscopeRenderer(256)
        hist = HistogramRenderer(256, 100)
        
        assert wf is not None
        assert vs is not None
        assert hist is not None
    
    def test_gpu_acceleration(self):
        """Test gpu.acceleration module."""
        from src.gpu.acceleration import GPUAccelerator
        
        gpu = GPUAccelerator()
        assert gpu is not None


class TestUIImports:
    """Test UI module imports (may skip if PySide6 not available)."""
    
    def test_ui_main_window(self):
        """Test ui.main_window module."""
        try:
            from src.ui.main_window import main, HAS_QT
            assert main is not None
        except ImportError:
            pytest.skip("PySide6 not installed")
    
    def test_ui_controls(self):
        """Test ui.controls module."""
        try:
            from src.ui.controls import CDLControls, ControlPanel, CurveEditor
            assert CDLControls is not None
        except ImportError:
            pytest.skip("UI controls require PySide6")
    
    def test_ui_scopes(self):
        """Test ui.scopes module."""
        try:
            from src.ui.scopes import ScopesPanel
            assert ScopesPanel is not None
        except ImportError:
            pytest.skip("UI scopes require PySide6")
    
    def test_ui_viewport(self):
        """Test ui.viewport module."""
        try:
            from src.ui.viewport import GPUViewport
            assert GPUViewport is not None
        except ImportError:
            pytest.skip("UI viewport require PySide6")
    
    def test_ui_workflow(self):
        """Test ui.workflow module."""
        from src.ui.workflow import WorkflowManager, Project
        
        assert WorkflowManager is not None
        assert Project is not None


class TestCrossModuleIntegration:
    """Test that modules can work together (integration tests)."""
    
    def test_color_pipeline_integration(self):
        """Test full color pipeline works without import errors."""
        import numpy as np
        from src.core.color_spaces import ColorSpaceConverter
        from src.core.color_science import ColorScience
        from src.core.transforms import ColorTransforms
        
        # Create test image
        image = np.random.rand(10, 10, 3).astype(np.float64)
        
        # Apply transforms
        converter = ColorSpaceConverter()
        xyz = converter.rgb_to_xyz(image)
        lab = converter.rgb_to_lab(image)
        
        # Apply CDL
        result = ColorTransforms.apply_cdl(
            image, 
            slope=(1.1, 1.0, 0.9),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=1.0
        )
        
        assert result.shape == image.shape
    
    def test_matching_pipeline_integration(self):
        """Test matching pipeline imports work together."""
        import numpy as np
        from src.engines.parametric.solver import ParametricSolver
        from src.engines.parametric.parameters import ParametricParameters
        from src.io.export import LUTExporter
        
        # Create solver
        solver = ParametricSolver()
        params = ParametricParameters.identity()
        
        # Apply to test data
        image = np.random.rand(10, 10, 3).astype(np.float64)
        result = solver.apply(image, params)
        
        assert result.shape == image.shape
    
    def test_export_pipeline_integration(self):
        """Test export functionality works."""
        import tempfile
        import os
        from src.engines.parametric.parameters import ParametricParameters, CDLParameters
        from src.io.export import LUTExporter, CDLExporter
        
        params = ParametricParameters.identity()
        cdl = CDLParameters()
        
        lut_exp = LUTExporter()
        cdl_exp = CDLExporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export LUT
            lut_path = os.path.join(tmpdir, 'test.cube')
            lut_exp.export_cube(lut_path, params, size=9)
            assert os.path.exists(lut_path)
            
            # Export CDL
            cdl_path = os.path.join(tmpdir, 'test.cdl')
            cdl_exp.export_cdl(cdl_path, cdl)
            assert os.path.exists(cdl_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
