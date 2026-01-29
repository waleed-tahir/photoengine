"""
Comprehensive Tests for GPU Acceleration and Integration Modules

Enterprise-grade testing for CUDA acceleration, API, plugins, and cloud processing.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGPUAccelerator:
    """Comprehensive tests for GPU acceleration."""
    
    @pytest.fixture
    def accelerator(self):
        from src.gpu.acceleration import GPUAccelerator
        return GPUAccelerator(backend='auto')
    
    def test_backend_selection(self, accelerator):
        """Test backend is selected."""
        assert accelerator.backend in ['cupy', 'numba', 'cpu']
    
    def test_identity_matrix_transform(self, accelerator):
        """Test identity matrix preserves image."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        matrix = np.eye(3, 4).astype(np.float32)
        
        result = accelerator.apply_matrix(image, matrix)
        
        assert_allclose(result, image, atol=1e-5)
    
    def test_matrix_transform_shape(self, accelerator):
        """Test matrix transform preserves shape."""
        image = np.random.rand(50, 75, 3).astype(np.float32)
        matrix = np.random.rand(3, 4).astype(np.float32)
        
        result = accelerator.apply_matrix(image, matrix)
        
        assert result.shape == image.shape
    
    def test_3d_lut_identity(self, accelerator):
        """Test identity 3D LUT preserves image."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(50, 50, 3).astype(np.float32)
        lut = ColorTransforms.create_identity_3d_lut(17)
        
        # Get the values array from the LUT3D object
        result = accelerator.apply_3d_lut(image, lut.values.astype(np.float32))
        
        # Allow for interpolation error
        assert_allclose(result, image, atol=0.05)
    
    def test_3d_lut_output_range(self, accelerator):
        """Test 3D LUT output is valid."""
        from src.core.transforms import ColorTransforms
        
        image = np.random.rand(50, 50, 3).astype(np.float32)
        lut = ColorTransforms.create_identity_3d_lut(17)
        
        result = accelerator.apply_3d_lut(image, lut.values.astype(np.float32))
        
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
    
    def test_to_gpu_from_gpu(self, accelerator):
        """Test GPU memory transfer."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        gpu_array = accelerator.to_gpu(image)
        back = accelerator.from_gpu(gpu_array)
        
        assert_allclose(back, image, atol=1e-6)


class TestCUDAKernels:
    """Tests for CUDA kernel implementations."""
    
    def test_cuda_processor_init(self):
        """Test CUDA processor initialization."""
        from src.gpu.cuda_kernels import CUDAProcessor
        
        processor = CUDAProcessor()
        assert processor is not None
    
    def test_lut_application(self):
        """Test LUT application via CUDA or fallback."""
        from src.gpu.cuda_kernels import CUDAProcessor
        from src.core.transforms import ColorTransforms
        
        processor = CUDAProcessor()
        
        image = np.random.rand(50, 50, 3).astype(np.float32)
        lut = ColorTransforms.create_identity_3d_lut(17)
        
        result = processor.apply_lut(image, lut.values.astype(np.float32))
        
        assert result.shape == image.shape
        assert_allclose(result, image, atol=0.05)
    
    def test_matrix_application(self):
        """Test matrix transform via CUDA or fallback."""
        from src.gpu.cuda_kernels import CUDAProcessor
        
        processor = CUDAProcessor()
        
        image = np.random.rand(50, 50, 3).astype(np.float32)
        matrix = np.eye(3, 4).astype(np.float32)
        
        result = processor.apply_matrix(image, matrix)
        
        assert_allclose(result, image, atol=1e-5)
    
    def test_cdl_application(self):
        """Test CDL via CUDA or fallback."""
        from src.gpu.cuda_kernels import CUDAProcessor
        
        processor = CUDAProcessor()
        
        image = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        
        result = processor.apply_cdl(
            image,
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=1.0
        )
        
        assert_allclose(result, image, atol=1e-5)


class TestAPIEndpoints:
    """Tests for REST API functionality."""
    
    @pytest.fixture
    def api(self):
        try:
            from src.integration.api import ChromaticaAPI
            return ChromaticaAPI()
        except ImportError:
            pytest.skip("API module requires optional dependencies")
    
    def test_health_check(self, api):
        """Test health check returns valid response."""
        health = api.health_check()
        
        assert health['status'] == 'healthy'
        assert 'version' in health
        assert 'supported_strategies' in health
    
    def test_supported_cameras(self, api):
        """Test camera list endpoint."""
        cameras = api.get_supported_cameras()
        
        assert len(cameras) > 0
        assert 'ARRI_LogC4_Alexa35' in cameras or 'sRGB' in cameras


class TestPluginSystem:
    """Tests for plugin system."""
    
    def test_plugin_manager_init(self):
        """Test plugin manager initialization."""
        from src.integration.plugins import PluginManager
        
        manager = PluginManager()
        assert manager is not None
    
    def test_list_plugins_empty(self):
        """Test listing plugins when none loaded."""
        from src.integration.plugins import PluginManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PluginManager(tmpdir)
            plugins = manager.list_plugins()
            
            assert isinstance(plugins, list)
    
    def test_builtin_transform_plugin(self):
        """Test built-in identity transform plugin."""
        from src.integration.plugins import IdentityTransform
        
        plugin = IdentityTransform()
        assert plugin.initialize()
        
        info = plugin.get_info()
        assert info.name == 'identity'
        assert info.plugin_type == 'transform'
    
    def test_builtin_dctl_exporter(self):
        """Test built-in DCTL exporter plugin."""
        from src.integration.plugins import DCTLExporter
        
        plugin = DCTLExporter()
        assert plugin.initialize()
        
        info = plugin.get_info()
        assert info.plugin_type == 'exporter'
        assert plugin.get_extension() == '.dctl'
    
    def test_identity_transform_apply(self):
        """Test identity transform application."""
        from src.integration.plugins import IdentityTransform
        
        plugin = IdentityTransform()
        plugin.initialize()
        
        image = np.random.rand(50, 50, 3)
        result = plugin.apply(image, {})
        
        assert_allclose(result, image)


class TestCloudIntegration:
    """Tests for cloud processing."""
    
    def test_local_storage_upload_download(self):
        """Test local storage upload/download."""
        from src.integration.cloud import LocalCloudStorage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalCloudStorage(tmpdir)
            
            # Create test file
            test_content = b"test content"
            local_file = os.path.join(tmpdir, "local_test.txt")
            with open(local_file, 'wb') as f:
                f.write(test_content)
            
            # Upload
            assert storage.upload(local_file, "remote/test.txt")
            
            # Check exists
            assert storage.exists("remote/test.txt")
            
            # Download
            download_path = os.path.join(tmpdir, "downloaded.txt")
            assert storage.download("remote/test.txt", download_path)
            
            with open(download_path, 'rb') as f:
                assert f.read() == test_content
    
    def test_cloud_processor_job_submission(self):
        """Test job submission (local mode)."""
        from src.integration.cloud import CloudProcessor, LocalCloudStorage
        from src.io.formats import ImageIO
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalCloudStorage(tmpdir)
            processor = CloudProcessor(storage)
            
            # Create test images
            source = np.random.rand(30, 30, 3).astype(np.float32)
            reference = np.random.rand(30, 30, 3).astype(np.float32)
            
            source_path = os.path.join(tmpdir, 'source.png')
            ref_path = os.path.join(tmpdir, 'reference.png')
            
            ImageIO.write(source_path, source)
            ImageIO.write(ref_path, reference)
            
            # Submit job
            job_id = processor.submit_job(source_path, ref_path)
            
            assert job_id is not None
            assert len(job_id) == 12
    
    def test_job_status_retrieval(self):
        """Test job status can be retrieved."""
        from src.integration.cloud import CloudProcessor, LocalCloudStorage
        from src.io.formats import ImageIO
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalCloudStorage(tmpdir)
            processor = CloudProcessor(storage)
            
            source = np.random.rand(30, 30, 3).astype(np.float32)
            reference = np.random.rand(30, 30, 3).astype(np.float32)
            
            source_path = os.path.join(tmpdir, 'source.png')
            ref_path = os.path.join(tmpdir, 'reference.png')
            
            ImageIO.write(source_path, source)
            ImageIO.write(ref_path, reference)
            
            job_id = processor.submit_job(source_path, ref_path)
            status = processor.get_job_status(job_id)
            
            assert status is not None
            assert status['job_id'] == job_id
            assert status['status'] == 'pending'


class TestWorkflowManager:
    """Tests for workflow management."""
    
    def test_project_creation(self):
        """Test project creation."""
        from src.ui.workflow import WorkflowManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkflowManager(tmpdir)
            
            project = manager.create_project("test_project")
            
            assert project.name == "test_project"
            assert manager.current_project == project
    
    def test_project_save_load(self):
        """Test project save and load."""
        from src.ui.workflow import WorkflowManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkflowManager(tmpdir)
            
            # Create and save
            project = manager.create_project("roundtrip_test")
            project.source_folder = "/path/to/source"
            manager.save_project(project)
            
            # Load
            loaded = manager.load_project("roundtrip_test")
            
            assert loaded.name == "roundtrip_test"
            assert loaded.source_folder == "/path/to/source"
    
    def test_preset_save_load(self):
        """Test preset save and load."""
        from src.ui.workflow import WorkflowManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkflowManager(tmpdir)
            
            params = {'slope': (1.1, 1.0, 0.9), 'saturation': 1.2}
            manager.save_preset('my_preset', params)
            
            loaded = manager.load_preset('my_preset')
            
            # Compare values (JSON may convert tuples to lists)
            assert list(loaded['slope']) == list(params['slope'])
            assert loaded['saturation'] == params['saturation']
    
    def test_list_projects(self):
        """Test project listing."""
        from src.ui.workflow import WorkflowManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkflowManager(tmpdir)
            
            manager.create_project("project_a")
            manager.create_project("project_b")
            
            projects = manager.list_projects()
            
            assert "project_a" in projects
            assert "project_b" in projects


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
