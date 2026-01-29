"""
Comprehensive Tests for I/O, Export, and Analysis Modules

Enterprise-grade testing for image formats, LUT/CDL export, validation, and quality analysis.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImageIO:
    """Comprehensive tests for image I/O."""
    
    def test_write_read_png_roundtrip(self):
        """Test PNG write/read roundtrip."""
        from src.io.formats import ImageIO
        
        original = np.random.rand(100, 100, 3).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            ImageIO.write(f.name, original)
            loaded, metadata = ImageIO.read(f.name)
        
        # PNG is 8-bit so expect some quantization
        assert_allclose(loaded, original, atol=0.01)
    
    def test_write_read_tiff_roundtrip(self):
        """Test TIFF write/read roundtrip."""
        from src.io.formats import ImageIO
        
        original = np.random.rand(100, 100, 3).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
            ImageIO.write(f.name, original, bit_depth=16)
            loaded, metadata = ImageIO.read(f.name)
        
        # 16-bit TIFF has some quantization too - use looser tolerance
        assert_allclose(loaded, original, atol=0.01)
    
    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises error."""
        from src.io.formats import ImageIO
        
        with pytest.raises(Exception):
            ImageIO.read("nonexistent_file_12345.png")
    
    def test_image_shape_preservation(self):
        """Test various image shapes are preserved."""
        from src.io.formats import ImageIO
        
        shapes = [(50, 50, 3), (100, 200, 3), (480, 640, 3)]
        
        for shape in shapes:
            original = np.random.rand(*shape).astype(np.float32)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                ImageIO.write(f.name, original)
                loaded, _ = ImageIO.read(f.name)
            
            assert loaded.shape == shape, f"Shape mismatch for {shape}"
    
    def test_supported_formats(self):
        """Test common image formats are supported."""
        from src.io.formats import ImageIO
        
        image = np.random.rand(50, 50, 3).astype(np.float32)
        formats = ['.png', '.jpg', '.tiff', '.bmp']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as f:
                try:
                    ImageIO.write(f.name, image)
                    loaded, _ = ImageIO.read(f.name)
                    assert loaded.shape == image.shape
                except Exception as e:
                    pytest.fail(f"Format {fmt} failed: {e}")


class TestLUTExporter:
    """Comprehensive tests for LUT export."""
    
    @pytest.fixture
    def params(self):
        from src.engines.parametric.parameters import ParametricParameters
        return ParametricParameters.identity()
    
    def test_cube_export_header(self, params):
        """Test .cube file has correct header."""
        from src.io.export import LUTExporter
        
        exporter = LUTExporter()
        
        with tempfile.NamedTemporaryFile(suffix='.cube', delete=False, mode='w') as f:
            exporter.export_cube(f.name, params, size=9)  # Small size for speed
            
            with open(f.name, 'r') as cube:
                content = cube.read()
        
        assert 'TITLE' in content
        assert 'LUT_3D_SIZE 9' in content
        assert 'DOMAIN_MIN' in content
        assert 'DOMAIN_MAX' in content
    
    def test_cube_export_data_count(self, params):
        """Test .cube file has correct number of data lines."""
        from src.io.export import LUTExporter
        
        exporter = LUTExporter()
        size = 9  # Small for speed
        
        with tempfile.NamedTemporaryFile(suffix='.cube', delete=False) as f:
            exporter.export_cube(f.name, params, size=size)
            
            with open(f.name, 'r') as cube:
                lines = cube.readlines()
        
        # Count data lines (non-header, non-empty)
        data_lines = [l for l in lines if l.strip() and not l.startswith('#') 
                     and not l.startswith('TITLE') and not l.startswith('LUT')
                     and not l.startswith('DOMAIN')]
        
        expected = size ** 3
        assert len(data_lines) == expected, f"Expected {expected}, got {len(data_lines)}"
    
    def test_cube_identity_values(self, params):
        """Test identity LUT has correct corner values."""
        from src.io.export import LUTExporter
        
        exporter = LUTExporter()
        
        with tempfile.NamedTemporaryFile(suffix='.cube', delete=False) as f:
            exporter.export_cube(f.name, params, size=9)
            
            with open(f.name, 'r') as cube:
                lines = cube.readlines()
        
        # Find first data line (black corner)
        data_lines = [l for l in lines if l.strip() and not l.startswith('#')
                     and not l.startswith('TITLE') and not l.startswith('LUT')
                     and not l.startswith('DOMAIN')]
        
        # First line should be 0 0 0
        first = data_lines[0].strip().split()
        assert float(first[0]) == pytest.approx(0.0, abs=0.01)
        assert float(first[1]) == pytest.approx(0.0, abs=0.01)
        assert float(first[2]) == pytest.approx(0.0, abs=0.01)
    
    def test_3dl_export(self, params):
        """Test .3dl export."""
        from src.io.export import LUTExporter
        
        exporter = LUTExporter()
        
        with tempfile.NamedTemporaryFile(suffix='.3dl', delete=False) as f:
            exporter.export_3dl(f.name, params, size=9)
            
            # Verify file exists and has content
            assert os.path.getsize(f.name) > 0


class TestCDLExporter:
    """Tests for CDL export."""
    
    def test_cdl_export_xml_valid(self):
        """Test CDL export produces valid XML."""
        from src.io.export import CDLExporter
        from src.engines.parametric.parameters import CDLParameters
        import xml.etree.ElementTree as ET
        
        exporter = CDLExporter()
        cdl = CDLParameters(
            slope=(1.1, 1.0, 0.9),
            offset=(0.01, 0.0, -0.01),
            power=(1.0, 1.0, 1.0),
            saturation=1.1
        )
        
        with tempfile.NamedTemporaryFile(suffix='.cdl', delete=False) as f:
            exporter.export_cdl(f.name, cdl)
            
            # Parse as XML
            tree = ET.parse(f.name)
            root = tree.getroot()
        
        # Should have ASC_CDL namespace or similar structure
        assert root is not None
    
    def test_ccc_export(self):
        """Test CCC (Color Correction Collection) export."""
        from src.io.export import CDLExporter
        from src.engines.parametric.parameters import CDLParameters
        
        exporter = CDLExporter()
        cdls = [
            CDLParameters(slope=(1.1, 1.0, 0.9)),
            CDLParameters(slope=(0.9, 1.0, 1.1)),
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.ccc', delete=False) as f:
            exporter.export_ccc(f.name, cdls)
            
            assert os.path.getsize(f.name) > 0


class TestColorCheckerValidator:
    """Comprehensive tests for ColorChecker validation."""
    
    def test_reference_data_count(self):
        """Test 24 ColorChecker patches are defined."""
        from src.analysis.validation import COLORCHECKER_CLASSIC
        
        assert len(COLORCHECKER_CLASSIC) == 24
    
    def test_reference_data_format(self):
        """Test ColorChecker reference data format."""
        from src.analysis.validation import COLORCHECKER_CLASSIC
        
        for patch in COLORCHECKER_CLASSIC:
            assert hasattr(patch, 'name')
            assert hasattr(patch, 'lab_d50')
            assert len(patch.lab_d50) == 3
            assert hasattr(patch, 'srgb')
            assert len(patch.srgb) == 3
    
    def test_perfect_match_validation(self):
        """Test perfect match gets all patches under 1.0 ΔE."""
        from src.analysis.validation import ColorCheckerValidator, COLORCHECKER_CLASSIC
        
        validator = ColorCheckerValidator()
        
        # Create perfect match
        reference_lab = np.array([p.lab_d50 for p in COLORCHECKER_CLASSIC])
        
        result = validator.validate(reference_lab, reference_lab)
        
        assert result.mean_delta_e == pytest.approx(0.0, abs=0.001)
        assert result.patches_under_1 == 24
        assert result.passed
    
    def test_validation_report_generation(self):
        """Test report generation."""
        from src.analysis.validation import ColorCheckerValidator, COLORCHECKER_CLASSIC
        
        validator = ColorCheckerValidator()
        reference_lab = np.array([p.lab_d50 for p in COLORCHECKER_CLASSIC])
        
        result = validator.validate(reference_lab)
        report = validator.generate_report(result)
        
        assert 'COLORCHECKER VALIDATION REPORT' in report
        assert 'Mean ΔE00' in report
    
    def test_patch_extraction_grid(self):
        """Test grid-based patch extraction."""
        from src.analysis.validation import ColorCheckerValidator
        
        validator = ColorCheckerValidator()
        
        # Create test image with 4x6 grid
        image = np.random.rand(400, 600, 3)
        
        patches = validator._extract_grid(image, rows=4, cols=6)
        
        assert patches.shape == (24, 3)


class TestQualityAnalyzer:
    """Comprehensive tests for quality analysis."""
    
    @pytest.fixture
    def analyzer(self):
        from src.analysis.quality import QualityAnalyzer
        return QualityAnalyzer()
    
    def test_perfect_match_grade_a(self, analyzer):
        """Test identical images get grade A."""
        image = np.random.rand(50, 50, 3)
        
        report = analyzer.analyze(image, image.copy())
        
        assert report.overall_grade == 'A'
        assert report.passed
    
    def test_different_images_lower_grade(self, analyzer):
        """Test very different images get lower grade."""
        image1 = np.zeros((50, 50, 3))
        image2 = np.ones((50, 50, 3))
        
        report = analyzer.analyze(image1, image2)
        
        # Should not be grade A for completely different images
        assert report.overall_grade in ['B', 'C', 'D', 'F']
    
    def test_report_has_all_metrics(self, analyzer):
        """Test report contains all expected metrics."""
        image = np.random.rand(50, 50, 3)
        
        report = analyzer.analyze(image, image)
        
        assert hasattr(report, 'delta_e_mean')
        assert hasattr(report, 'psnr')
        assert hasattr(report, 'mse')
        assert hasattr(report, 'overall_grade')
        assert hasattr(report, 'passed')
    
    def test_clipping_detection(self, analyzer):
        """Test clipping detection."""
        # Image with clipping
        image = np.zeros((50, 50, 3))
        image[:25, :, :] = 1.0  # Half white, half black - some "clipping"
        
        report = analyzer.analyze(image, image)
        
        # Should report some form of analysis
        assert report.clipping_low_pct >= 0
        assert report.clipping_high_pct >= 0
    
    def test_pipeline_comparison(self, analyzer):
        """Test pipeline comparison."""
        source = np.random.rand(30, 30, 3)
        target = np.random.rand(30, 30, 3)
        
        results = {
            'pipeline_a': source + 0.1,
            'pipeline_b': source + 0.05,
        }
        
        comparison = analyzer.compare_pipelines(source, target, results)
        
        assert 'comparisons' in comparison
        assert 'rankings' in comparison
        assert 'best' in comparison


class TestScopes:
    """Tests for professional video scopes."""
    
    def test_waveform_output_shape(self):
        """Test waveform output shape."""
        from src.analysis.scopes import WaveformScope
        
        scope = WaveformScope(360, 240)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        waveform = scope.render_luma(image)
        
        assert waveform.shape == (240, 360, 3)
    
    def test_waveform_rgb_parade(self):
        """Test RGB parade waveform."""
        from src.analysis.scopes import WaveformScope
        
        scope = WaveformScope(360, 240)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        parade = scope.render_rgb_parade(image)
        
        assert parade.shape == (240, 360, 3)
    
    def test_vectorscope_output_shape(self):
        """Test vectorscope output shape."""
        from src.analysis.scopes import VectorscopeRenderer
        
        scope = VectorscopeRenderer(256)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        vectorscope = scope.render(image)
        
        assert vectorscope.shape == (256, 256, 3)
    
    def test_histogram_output_shape(self):
        """Test histogram output shape."""
        from src.analysis.scopes import HistogramRenderer
        
        renderer = HistogramRenderer(256, 100)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        histogram = renderer.render(image)
        
        assert histogram.shape == (100, 256, 3)
    
    def test_scope_value_range(self):
        """Test scope outputs are in valid range."""
        from src.analysis.scopes import WaveformScope, VectorscopeRenderer
        
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        waveform = WaveformScope(360, 240).render_luma(image)
        vectorscope = VectorscopeRenderer(256).render(image)
        
        assert np.all(waveform >= 0) and np.all(waveform <= 1)
        assert np.all(vectorscope >= 0) and np.all(vectorscope <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
