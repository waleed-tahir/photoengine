"""
GPU Acceleration Interface

Unified interface for CUDA, Metal, and OpenCL backends.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Check available backends
HAS_CUPY = False
HAS_NUMBA = False

try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("CuPy CUDA backend available")
except ImportError:
    pass

try:
    from numba import cuda, jit
    HAS_NUMBA = True
    logger.info("Numba JIT available")
except ImportError:
    pass


class GPUAccelerator:
    """
    Unified GPU acceleration interface.
    
    Automatically selects best available backend:
    1. cupy (CUDA) - fastest for NVIDIA GPUs
    2. numba CUDA - fallback CUDA
    3. CPU (numpy) - always available
    
    Example:
        >>> gpu = GPUAccelerator()
        >>> result = gpu.apply_3d_lut(image, lut)
    """
    
    def __init__(self, backend: str = 'auto'):
        if backend == 'auto':
            if HAS_CUPY:
                self.backend = 'cupy'
            elif HAS_NUMBA and cuda.is_available():
                self.backend = 'numba'
            else:
                self.backend = 'cpu'
        else:
            self.backend = backend
        
        logger.info(f"GPU accelerator using backend: {self.backend}")
    
    def apply_3d_lut(self, image: npt.NDArray, lut: npt.NDArray) -> npt.NDArray:
        """
        Apply 3D LUT with GPU acceleration.
        
        Args:
            image: Input [H, W, 3] in 0-1 range
            lut: 3D LUT [size, size, size, 3]
            
        Returns:
            Transformed image
        """
        if self.backend == 'cupy' and HAS_CUPY:
            return self._cupy_apply_3d_lut(image, lut)
        else:
            return self._cpu_apply_3d_lut(image, lut)
    
    def _cupy_apply_3d_lut(self, image: npt.NDArray, lut: npt.NDArray) -> npt.NDArray:
        """CuPy CUDA implementation of 3D LUT."""
        d_image = cp.asarray(image, dtype=cp.float32)
        d_lut = cp.asarray(lut, dtype=cp.float32)
        
        h, w = d_image.shape[:2]
        size = d_lut.shape[0]
        
        # Scale to LUT coordinates
        coords = d_image * (size - 1)
        coords = cp.clip(coords, 0, size - 1 - 1e-6)
        
        # Integer and fractional parts
        coords_floor = cp.floor(coords).astype(cp.int32)
        coords_frac = coords - coords_floor
        
        # Trilinear interpolation
        r0, g0, b0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
        r1 = cp.minimum(r1, size - 1)
        g1 = cp.minimum(g1, size - 1)
        b1 = cp.minimum(b1, size - 1)
        
        dr = coords_frac[..., 0:1]
        dg = coords_frac[..., 1:2]
        db = coords_frac[..., 2:3]
        
        c000 = d_lut[r0, g0, b0]
        c001 = d_lut[r0, g0, b1]
        c010 = d_lut[r0, g1, b0]
        c011 = d_lut[r0, g1, b1]
        c100 = d_lut[r1, g0, b0]
        c101 = d_lut[r1, g0, b1]
        c110 = d_lut[r1, g1, b0]
        c111 = d_lut[r1, g1, b1]
        
        result = (c000 * (1-dr)*(1-dg)*(1-db) +
                 c001 * (1-dr)*(1-dg)*db +
                 c010 * (1-dr)*dg*(1-db) +
                 c011 * (1-dr)*dg*db +
                 c100 * dr*(1-dg)*(1-db) +
                 c101 * dr*(1-dg)*db +
                 c110 * dr*dg*(1-db) +
                 c111 * dr*dg*db)
        
        return cp.asnumpy(result)
    
    def _cpu_apply_3d_lut(self, image: npt.NDArray, lut: npt.NDArray) -> npt.NDArray:
        """CPU fallback implementation."""
        from scipy.interpolate import RegularGridInterpolator
        
        size = lut.shape[0]
        x = np.linspace(0, 1, size)
        
        interp = RegularGridInterpolator((x, x, x), lut, method='linear')
        
        points = np.clip(image.reshape(-1, 3), 0, 1)
        result = interp(points)
        
        return result.reshape(image.shape)
    
    def apply_matrix(self, image: npt.NDArray, matrix: npt.NDArray) -> npt.NDArray:
        """Apply 3x3 or 3x4 matrix transform with GPU."""
        if self.backend == 'cupy' and HAS_CUPY:
            d_image = cp.asarray(image, dtype=cp.float32)
            d_matrix = cp.asarray(matrix, dtype=cp.float32)
            
            flat = d_image.reshape(-1, 3)
            if matrix.shape[1] == 4:
                result = cp.dot(flat, d_matrix[:, :3].T) + d_matrix[:, 3]
            else:
                result = cp.dot(flat, d_matrix.T)
            
            return cp.asnumpy(result.reshape(image.shape))
        else:
            flat = image.reshape(-1, 3)
            if matrix.shape[1] == 4:
                result = np.dot(flat, matrix[:, :3].T) + matrix[:, 3]
            else:
                result = np.dot(flat, matrix.T)
            return result.reshape(image.shape)
    
    def batch_process(self, images: list, operation: callable,
                      batch_size: int = 4) -> list:
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = [operation(img) for img in batch]
            results.extend(batch_results)
        return results
    
    def to_gpu(self, array: npt.NDArray) -> npt.NDArray:
        """Transfer array to GPU memory."""
        if self.backend == 'cupy' and HAS_CUPY:
            return cp.asarray(array)
        return array  # CPU fallback returns same array
    
    def from_gpu(self, array) -> npt.NDArray:
        """Transfer array from GPU to CPU memory."""
        if self.backend == 'cupy' and HAS_CUPY:
            return cp.asnumpy(array)
        return np.asarray(array)

