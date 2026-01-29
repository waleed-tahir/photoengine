"""
Comprehensive Tests for Core Color Science Module

Enterprise-grade testing for color_spaces, transforms, color_science, and ACES pipeline.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestColorSpaceConverter:
    """Comprehensive tests for ColorSpaceConverter."""
    
    @pytest.fixture
    def converter(self):
        from src.core.color_spaces import ColorSpaceConverter
        return ColorSpaceConverter()
    
    # ==================== RGB to XYZ Tests ====================
    
    def test_rgb_to_xyz_black(self, converter):
        """Test black point conversion."""
        black = np.array([[[0.0, 0.0, 0.0]]])
        xyz = converter.rgb_to_xyz(black)
        assert_allclose(xyz, 0.0, atol=1e-10)
    
    def test_rgb_to_xyz_white_d65(self, converter):
        """Test white point converts to D65 white."""
        white = np.array([[[1.0, 1.0, 1.0]]])
        xyz = converter.rgb_to_xyz(white)
        # D65 white point is approximately [0.95047, 1.0, 1.08883]
        assert xyz[0, 0, 1] == pytest.approx(1.0, abs=0.01)
    
    def test_rgb_to_xyz_red(self, converter):
        """Test pure red conversion."""
        red = np.array([[[1.0, 0.0, 0.0]]])
        xyz = converter.rgb_to_xyz(red)
        # Red should have positive X, low Y, near-zero Z
        assert xyz[0, 0, 0] > 0.3
        assert xyz[0, 0, 2] < 0.1
    
    def test_rgb_to_xyz_shape_preservation(self, converter):
        """Test shape is preserved through conversion."""
        image = np.random.rand(100, 150, 3)
        xyz = converter.rgb_to_xyz(image)
        assert xyz.shape == image.shape
    
    def test_rgb_to_xyz_batch_consistency(self, converter):
        """Test batch conversion matches single pixel conversion."""
        pixel = np.array([0.5, 0.3, 0.7])
        batch = np.tile(pixel, (10, 10, 1))
        
        single = converter.rgb_to_xyz(pixel.reshape(1, 1, 3))
        batched = converter.rgb_to_xyz(batch)
        
        for i in range(10):
            for j in range(10):
                assert_allclose(batched[i, j], single[0, 0], atol=1e-10)
    
    # ==================== XYZ to LAB Tests ====================
    
    def test_xyz_to_lab_white(self, converter):
        """Test D65 white point converts to L*=100."""
        # D65 white point
        white_xyz = np.array([[[0.95047, 1.0, 1.08883]]])
        lab = converter.xyz_to_lab(white_xyz)
        assert lab[0, 0, 0] == pytest.approx(100.0, abs=0.1)
        assert lab[0, 0, 1] == pytest.approx(0.0, abs=0.5)
        assert lab[0, 0, 2] == pytest.approx(0.0, abs=0.5)
    
    def test_xyz_to_lab_black(self, converter):
        """Test black converts to L*=0."""
        black = np.array([[[0.0, 0.0, 0.0]]])
        lab = converter.xyz_to_lab(black)
        assert lab[0, 0, 0] == pytest.approx(0.0, abs=0.1)
    
    def test_xyz_to_lab_mid_gray(self, converter):
        """Test 18% gray (L*=50 approximately)."""
        # 18% reflectance
        gray_xyz = np.array([[[0.18 * 0.95047, 0.18, 0.18 * 1.08883]]])
        lab = converter.xyz_to_lab(gray_xyz)
        assert 45 < lab[0, 0, 0] < 55
    
    # ==================== RGB to LAB Roundtrip ====================
    
    def test_rgb_lab_roundtrip(self, converter):
        """Test RGB -> LAB -> RGB roundtrip."""
        original = np.random.rand(50, 50, 3) * 0.9 + 0.05  # Avoid extremes
        
        lab = converter.rgb_to_lab(original)
        recovered = converter.lab_to_rgb(lab)
        
        assert_allclose(recovered, original, atol=1e-5)
    
    def test_rgb_xyz_roundtrip(self, converter):
        """Test RGB -> XYZ -> RGB roundtrip."""
        original = np.random.rand(50, 50, 3)
        
        xyz = converter.rgb_to_xyz(original)
        recovered = converter.xyz_to_rgb(xyz)
        
        assert_allclose(recovered, original, atol=1e-10)
    
    # ==================== HSV Tests ====================
    
    def test_rgb_to_hsv_red(self, converter):
        """Test pure red HSV values."""
        red = np.array([[[1.0, 0.0, 0.0]]])
        hsv = converter.rgb_to_hsv(red)
        assert hsv[0, 0, 0] == pytest.approx(0.0, abs=0.01)  # Hue = 0
        assert hsv[0, 0, 1] == pytest.approx(1.0, abs=0.01)  # Saturation = 1
        assert hsv[0, 0, 2] == pytest.approx(1.0, abs=0.01)  # Value = 1
    
    def test_rgb_to_hsv_green(self, converter):
        """Test pure green HSV values."""
        green = np.array([[[0.0, 1.0, 0.0]]])
        hsv = converter.rgb_to_hsv(green)
        assert hsv[0, 0, 0] == pytest.approx(120.0, abs=1.0)  # Hue = 120
    
    def test_rgb_to_hsv_blue(self, converter):
        """Test pure blue HSV values."""
        blue = np.array([[[0.0, 0.0, 1.0]]])
        hsv = converter.rgb_to_hsv(blue)
        assert hsv[0, 0, 0] == pytest.approx(240.0, abs=1.0)  # Hue = 240
    
    # ==================== Edge Cases ====================
    
    def test_out_of_gamut_handling(self, converter):
        """Test handling of out-of-gamut values."""
        out_of_gamut = np.array([[[1.5, -0.2, 0.5]]])
        xyz = converter.rgb_to_xyz(out_of_gamut)
        # Should not crash, may produce out of range XYZ
        assert not np.isnan(xyz).any()
    
    def test_very_small_values(self, converter):
        """Test numerical stability with very small values."""
        tiny = np.array([[[1e-10, 1e-10, 1e-10]]])
        lab = converter.rgb_to_lab(tiny)
        assert not np.isnan(lab).any()
        assert not np.isinf(lab).any()


class TestColorTransforms:
    """Comprehensive tests for ColorTransforms."""
    
    # ==================== Matrix Transform Tests ====================
    
    def test_identity_matrix_3x3(self):
        """Test identity matrix preserves image."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(50, 50, 3)
        identity = np.eye(3)
        
        result = ColorTransforms.apply_matrix_3x3(image, identity)
        assert_allclose(result, image, atol=1e-10)
    
    def test_identity_matrix_3x4(self):
        """Test 3x4 identity matrix preserves image."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(50, 50, 3)
        identity = np.eye(3, 4)
        
        result = ColorTransforms.apply_matrix_3x4(image, identity)
        assert_allclose(result, image, atol=1e-10)
    
    def test_offset_only_matrix(self):
        """Test matrix with only offset."""
        from src.core.transforms import ColorTransforms
        
        image = np.ones((10, 10, 3)) * 0.5
        matrix = np.eye(3, 4)
        matrix[:, 3] = [0.1, 0.2, 0.3]
        
        result = ColorTransforms.apply_matrix_3x4(image, matrix)
        
        assert_allclose(result[..., 0], 0.6, atol=1e-10)
        assert_allclose(result[..., 1], 0.7, atol=1e-10)
        assert_allclose(result[..., 2], 0.8, atol=1e-10)
    
    def test_scale_matrix(self):
        """Test diagonal scaling matrix."""
        from src.core.transforms import ColorTransforms
        
        image = np.ones((10, 10, 3)) * 0.5
        matrix = np.diag([2.0, 1.5, 0.5])
        
        result = ColorTransforms.apply_matrix_3x3(image, matrix)
        
        assert_allclose(result[..., 0], 1.0, atol=1e-10)
        assert_allclose(result[..., 1], 0.75, atol=1e-10)
        assert_allclose(result[..., 2], 0.25, atol=1e-10)
    
    # ==================== CDL Tests ====================
    
    def test_cdl_identity(self):
        """Test CDL with identity parameters."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(50, 50, 3)
        
        result = ColorTransforms.apply_cdl(
            image,
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=1.0
        )
        
        assert_allclose(result, image, atol=1e-10)
    
    def test_cdl_slope(self):
        """Test CDL slope (gain)."""
        from src.core.transforms import ColorTransforms
        
        image = np.ones((10, 10, 3)) * 0.5
        
        result = ColorTransforms.apply_cdl(
            image,
            slope=(2.0, 1.0, 0.5),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=1.0
        )
        
        assert_allclose(result[..., 0], 1.0, atol=1e-10)
        assert_allclose(result[..., 1], 0.5, atol=1e-10)
        assert_allclose(result[..., 2], 0.25, atol=1e-10)
    
    def test_cdl_offset(self):
        """Test CDL offset (lift)."""
        from src.core.transforms import ColorTransforms
        
        image = np.zeros((10, 10, 3))
        
        result = ColorTransforms.apply_cdl(
            image,
            slope=(1.0, 1.0, 1.0),
            offset=(0.1, 0.2, 0.3),
            power=(1.0, 1.0, 1.0),
            saturation=1.0
        )
        
        assert_allclose(result[..., 0], 0.1, atol=1e-10)
        assert_allclose(result[..., 1], 0.2, atol=1e-10)
        assert_allclose(result[..., 2], 0.3, atol=1e-10)
    
    def test_cdl_power(self):
        """Test CDL power (gamma)."""
        from src.core.transforms import ColorTransforms
        
        image = np.ones((10, 10, 3)) * 0.25
        
        result = ColorTransforms.apply_cdl(
            image,
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(0.5, 1.0, 2.0),
            saturation=1.0
        )
        
        assert result[0, 0, 0] == pytest.approx(0.5, abs=0.01)  # 0.25^0.5
        assert result[0, 0, 1] == pytest.approx(0.25, abs=0.01)  # 0.25^1.0
        assert result[0, 0, 2] == pytest.approx(0.0625, abs=0.01)  # 0.25^2.0
    
    def test_cdl_saturation_zero(self):
        """Test CDL saturation = 0 produces grayscale."""
        from src.core.transforms import ColorTransforms
        
        image = np.array([[[1.0, 0.0, 0.0]]])  # Pure red
        
        result = ColorTransforms.apply_cdl(
            image,
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=0.0
        )
        
        # All channels should be equal (grayscale)
        assert result[0, 0, 0] == pytest.approx(result[0, 0, 1], abs=0.01)
        assert result[0, 0, 1] == pytest.approx(result[0, 0, 2], abs=0.01)
    
    # ==================== Curve Tests ====================
    
    def test_apply_curve_identity(self):
        """Test identity curve preserves values."""
        from src.core.transforms import ColorTransforms, Curve
        
        image = np.random.rand(50, 50, 3)
        curve = Curve(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
        
        result = ColorTransforms.apply_curve(image, curve)
        
        assert_allclose(result, image, atol=0.01)
    
    # ==================== 3D LUT Tests ====================
    
    def test_identity_3d_lut(self):
        """Test identity 3D LUT."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(50, 50, 3)
        lut = ColorTransforms.create_identity_3d_lut(17)
        
        result = lut.apply(image)
        
        assert_allclose(result, image, atol=0.02)
    
    def test_3d_lut_object_creation(self):
        """Test 3D LUT creation returns LUT3D object."""
        from src.core.transforms import ColorTransforms, LUT3D
        
        lut = ColorTransforms.create_identity_3d_lut(33)
        assert isinstance(lut, LUT3D)
        assert lut.size == 33
    
    def test_3d_lut_values_shape(self):
        """Test 3D LUT values have correct shape."""
        from src.core.transforms import ColorTransforms
        
        lut = ColorTransforms.create_identity_3d_lut(17)
        assert lut.values.shape == (17, 17, 17, 3)
    
    def test_3d_lut_black_white_corners(self):
        """Test 3D LUT corner values."""
        from src.core.transforms import ColorTransforms
        
        lut = ColorTransforms.create_identity_3d_lut(17)
        
        # Black corner
        assert_allclose(lut.values[0, 0, 0], [0, 0, 0], atol=1e-10)
        # White corner
        assert_allclose(lut.values[-1, -1, -1], [1, 1, 1], atol=1e-10)
        # Red corner
        assert_allclose(lut.values[-1, 0, 0], [1, 0, 0], atol=1e-10)
    
    # ==================== Exposure/Contrast Tests ====================
    
    def test_exposure_positive(self):
        """Test positive exposure adjustment."""
        from src.core.transforms import ColorTransforms
        
        image = np.ones((10, 10, 3)) * 0.25
        
        result = ColorTransforms.apply_exposure(image, stops=1.0)
        
        assert_allclose(result, 0.5, atol=0.01)
    
    def test_exposure_negative(self):
        """Test negative exposure adjustment."""
        from src.core.transforms import ColorTransforms
        
        image = np.ones((10, 10, 3)) * 0.5
        
        result = ColorTransforms.apply_exposure(image, stops=-1.0)
        
        assert_allclose(result, 0.25, atol=0.01)
    
    def test_contrast_increase(self):
        """Test contrast increase."""
        from src.core.transforms import ColorTransforms
        
        image = np.array([[[0.18, 0.5, 0.75]]])  # Use 0.18 as pivot
        
        result = ColorTransforms.apply_contrast(image, contrast=2.0, pivot=0.18)
        
        # Pivot (0.18) should stay same, others move away
        assert result[0, 0, 0] == pytest.approx(0.18, abs=0.01)
        assert result[0, 0, 1] > 0.5  # Above pivot, moves up
        assert result[0, 0, 2] > 0.75  # Above pivot, moves up more


class TestDeltaE2000:
    """Comprehensive tests for ΔE2000 color difference."""
    
    def test_identical_colors(self):
        """Test ΔE2000 between identical colors is 0."""
        from src.core.color_science import ColorScience
        
        lab = np.array([[50.0, 10.0, -20.0]])
        
        delta_e = ColorScience.delta_e_2000(lab, lab)
        
        assert delta_e[0] == pytest.approx(0.0, abs=1e-10)
    
    def test_known_delta_e_values(self):
        """Test ΔE2000 against known reference values."""
        from src.core.color_science import ColorScience
        
        # Test cases from Sharma et al. 2005
        test_cases = [
            # (L1, a1, b1), (L2, a2, b2), expected_dE
            ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
            ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
            ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
        ]
        
        for (L1, a1, b1), (L2, a2, b2), expected in test_cases:
            lab1 = np.array([[L1, a1, b1]])
            lab2 = np.array([[L2, a2, b2]])
            
            result = ColorScience.delta_e_2000(lab1, lab2)
            
            assert result[0] == pytest.approx(expected, abs=0.01), \
                f"Expected {expected}, got {result[0]}"
    
    def test_delta_e_symmetry(self):
        """Test ΔE2000 is symmetric."""
        from src.core.color_science import ColorScience
        
        lab1 = np.array([[50.0, 20.0, 30.0]])
        lab2 = np.array([[60.0, -10.0, 40.0]])
        
        forward = ColorScience.delta_e_2000(lab1, lab2)
        backward = ColorScience.delta_e_2000(lab2, lab1)
        
        assert forward[0] == pytest.approx(backward[0], abs=1e-6)
    
    def test_delta_e_just_noticeable(self):
        """Test JND threshold (~1.0 ΔE2000)."""
        from src.core.color_science import ColorScience
        
        # Two colors that should be just noticeably different
        lab1 = np.array([[50.0, 0.0, 0.0]])
        lab2 = np.array([[51.0, 0.0, 0.0]])  # 1 L* unit difference
        
        result = ColorScience.delta_e_2000(lab1, lab2)
        
        assert 0.5 < result[0] < 1.5
    
    def test_delta_e_batch(self):
        """Test batch ΔE2000 calculation."""
        from src.core.color_science import ColorScience
        
        lab1 = np.random.rand(100, 3) * 100
        lab2 = np.random.rand(100, 3) * 100
        
        result = ColorScience.delta_e_2000(lab1, lab2)
        
        assert result.shape == (100,)
        assert np.all(result >= 0)


class TestACESPipeline:
    """Comprehensive tests for ACES 2.2 pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        from src.core.aces_pipeline import ACESPipeline
        return ACESPipeline()
    
    def test_idt_library_exists(self, pipeline):
        """Test IDT library is populated."""
        assert len(pipeline.idt_library) > 0
    
    def test_common_cameras_exist(self, pipeline):
        """Test common camera profiles exist."""
        expected = ['ARRI_LogC3_Alexa', 'ARRI_LogC4_Alexa35', 
                    'Sony_SLog3_SGamut3', 'RED_Log3G10_RWG', 'sRGB']
        
        for camera in expected:
            assert camera in pipeline.idt_library, f"Missing camera: {camera}"
    
    def test_idt_black_preservation(self, pipeline):
        """Test IDT preserves black for sRGB."""
        black = np.zeros((10, 10, 3))
        
        result = pipeline.apply_idt(black, 'sRGB')
        assert np.allclose(result, 0, atol=0.01)
    
    def test_idt_srgb_produces_valid_output(self, pipeline):
        """Test sRGB IDT produces valid output."""
        source = np.random.rand(50, 50, 3) * 0.8 + 0.1
        
        result = pipeline.apply_idt(source, 'sRGB')
        
        assert not np.isnan(result).any()
        assert result.shape == source.shape
    
    def test_lmt_identity(self, pipeline):
        """Test LMT with identity parameters preserves image."""
        from src.core.aces_pipeline import LMTParams
        
        image = np.random.rand(30, 30, 3) * 0.5 + 0.25
        lmt = LMTParams()  # All defaults (identity)
        
        result = pipeline.apply_lmt(image, lmt)
        
        assert_allclose(result, image, atol=1e-10)
    
    def test_lmt_cdl_application(self, pipeline):
        """Test LMT CDL parameters are applied."""
        from src.core.aces_pipeline import LMTParams, CDLParams
        
        gray = np.ones((10, 10, 3)) * 0.5
        
        # Apply red boost via CDL
        lmt = LMTParams(cdl=CDLParams(
            slope=(1.5, 1.0, 1.0),
            offset=(0, 0, 0),
            power=(1, 1, 1),
            saturation=1.0
        ))
        
        result = pipeline.apply_lmt(gray, lmt)
        
        assert result[0, 0, 0] > result[0, 0, 1]
        assert result[0, 0, 0] > result[0, 0, 2]
    
    def test_rrt_odt_srgb(self, pipeline):
        """Test RRT+ODT produces valid sRGB output."""
        image_ap0 = np.random.rand(30, 30, 3) * 0.3  # Scene-linear
        
        result = pipeline.apply_rrt_odt(image_ap0, 'sRGB')
        
        assert not np.isnan(result).any()
        assert np.all(result >= 0) and np.all(result <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
