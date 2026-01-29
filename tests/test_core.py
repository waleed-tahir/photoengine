"""
Unit Tests for Core Color Science
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose


class TestColorSpaces:
    """Test color space conversions."""
    
    def test_rgb_to_xyz_and_back(self):
        """Test round-trip RGB -> XYZ -> RGB."""
        from src.core.color_spaces import ColorSpaceConverter
        
        converter = ColorSpaceConverter()
        
        # Test with known sRGB values
        rgb = np.array([[[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [1.0, 1.0, 1.0]]])
        
        xyz = converter.rgb_to_xyz(rgb)
        rgb_back = converter.xyz_to_rgb(xyz)
        
        assert_allclose(rgb, rgb_back, atol=1e-6)
    
    def test_rgb_to_lab_and_back(self):
        """Test round-trip RGB -> LAB -> RGB."""
        from src.core.color_spaces import ColorSpaceConverter
        
        converter = ColorSpaceConverter()
        
        rgb = np.array([[[0.5, 0.5, 0.5]]])
        lab = converter.rgb_to_lab(rgb)
        rgb_back = converter.lab_to_rgb(lab)
        
        assert_allclose(rgb, rgb_back, atol=1e-5)
    
    def test_neutral_gray_lab(self):
        """Test that neutral gray has a=b=0 in LAB."""
        from src.core.color_spaces import ColorSpaceConverter
        
        converter = ColorSpaceConverter()
        
        gray = np.array([[[0.5, 0.5, 0.5]]])
        lab = converter.rgb_to_lab(gray)
        
        assert abs(lab[0, 0, 1]) < 0.1  # a should be ~0
        assert abs(lab[0, 0, 2]) < 0.1  # b should be ~0
    
    def test_white_point_lab(self):
        """Test that D65 white has L=100, a=b=0."""
        from src.core.color_spaces import ColorSpaceConverter
        
        converter = ColorSpaceConverter()
        
        white = np.array([[[1.0, 1.0, 1.0]]])
        lab = converter.rgb_to_lab(white)
        
        assert_allclose(lab[0, 0, 0], 100.0, atol=0.5)
        assert abs(lab[0, 0, 1]) < 0.1
        assert abs(lab[0, 0, 2]) < 0.1


class TestDeltaE2000:
    """Test ΔE2000 calculation."""
    
    def test_identical_colors(self):
        """Test that identical colors have ΔE = 0."""
        from src.core.color_science import ColorScience
        
        lab = np.array([[[50.0, 0.0, 0.0]]])
        delta_e = ColorScience.delta_e_2000(lab, lab)
        
        assert delta_e[0, 0] == 0.0
    
    def test_known_values(self):
        """Test against known ΔE2000 values."""
        from src.core.color_science import ColorScience
        
        # CIE test pair 1
        lab1 = np.array([50.0, 2.6772, -79.7751])
        lab2 = np.array([50.0, 0.0, -82.7485])
        
        delta_e = ColorScience.delta_e_2000(
            lab1.reshape(1, 3), 
            lab2.reshape(1, 3)
        )
        
        # Expected ~2.0425
        assert abs(delta_e[0] - 2.0425) < 0.01
    
    def test_perceptibility_threshold(self):
        """Test that small differences give ΔE < 1."""
        from src.core.color_science import ColorScience
        
        lab1 = np.array([50.0, 0.0, 0.0])
        lab2 = np.array([50.1, 0.05, 0.05])  # Very small difference
        
        delta_e = ColorScience.delta_e_2000(
            lab1.reshape(1, 3), 
            lab2.reshape(1, 3)
        )
        
        assert delta_e[0] < 1.0


class TestTransforms:
    """Test color transforms."""
    
    def test_identity_matrix(self):
        """Test that identity matrix doesn't change image."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(10, 10, 3)
        identity = np.eye(3)
        
        result = ColorTransforms.apply_matrix_3x3(image, identity)
        
        assert_allclose(image, result, atol=1e-10)
    
    def test_cdl_identity(self):
        """Test CDL with identity parameters."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(10, 10, 3)
        
        result = ColorTransforms.apply_cdl(
            image,
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=1.0
        )
        
        assert_allclose(image, result, atol=1e-10)
    
    def test_exposure_adjustment(self):
        """Test exposure adjustment."""
        from src.core.transforms import ColorTransforms
        
        image = np.full((10, 10, 3), 0.18)  # 18% gray
        
        # +1 stop should double the value
        result = ColorTransforms.apply_exposure(image, 1.0)
        
        assert_allclose(result, 0.36, atol=0.01)
    
    def test_3d_lut_identity(self):
        """Test that identity 3D LUT doesn't change image."""
        from src.core.transforms import ColorTransforms, LUT3D
        
        lut = ColorTransforms.create_identity_3d_lut(17)
        image = np.random.rand(10, 10, 3)
        
        result = lut.apply(image)
        
        assert_allclose(image, result, atol=0.01)


class TestParametricSolver:
    """Test parametric solver."""
    
    def test_identity_match(self):
        """Test matching identical images returns near-identity."""
        try:
            from src.engines.parametric.solver import ParametricSolver
            from src.engines.parametric.parameters import ParametricParameters
        except ImportError:
            pytest.skip("Solver requires dependencies")
        
        solver = ParametricSolver()
        
        # Create identical test images
        image = np.random.rand(30, 30, 3).astype(np.float64)
        
        result = solver.match(image, image.copy())
        
        # Should match well (may not be perfect due to optimization)
        assert result['metrics']['delta_e_mean'] < 2.0
    
    def test_cdl_optimization(self):
        """Test CDL-only optimization."""
        from src.engines.parametric.solver import ParametricSolver
        
        solver = ParametricSolver()
        
        source = np.random.rand(30, 30, 3).astype(np.float64) * 0.5
        target = source * 1.2 + 0.05  # Simple linear transform
        target = np.clip(target, 0, 1)
        
        result = solver.optimize_cdl_only(source, target)
        
        # Should achieve reasonable match
        assert result['metrics']['delta_e_mean'] < 5.0


class TestACESPipeline:
    """Test ACES pipeline."""
    
    def test_input_device_transform(self):
        """Test IDT with known camera profile."""
        from src.core.aces_pipeline import ACESPipeline
        
        pipeline = ACESPipeline()
        
        # Test with sRGB input (use correct case)
        srgb_image = np.random.rand(10, 10, 3).astype(np.float64)
        
        aces_ap0 = pipeline.apply_idt(srgb_image, 'sRGB')
        
        # Should produce valid output
        assert aces_ap0.shape == srgb_image.shape
        assert not np.isnan(aces_ap0).any()
    
    def test_full_pipeline(self):
        """Test full ACES pipeline: IDT -> LMT -> RRT+ODT."""
        from src.core.aces_pipeline import ACESPipeline, LMTParams, CDLParams
        
        pipeline = ACESPipeline()
        
        # Create test image
        source = np.random.rand(20, 20, 3).astype(np.float64) * 0.5
        
        # Process through pipeline step by step
        ap0 = pipeline.apply_idt(source, 'sRGB')
        lmt = LMTParams(cdl=CDLParams())
        graded = pipeline.apply_lmt(ap0, lmt)
        result = pipeline.apply_rrt_odt(graded, 'sRGB')
        
        # Should produce valid, bounded output
        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1.01)  # Allow small overshoot


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
