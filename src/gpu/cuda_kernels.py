"""
CUDA Kernels for Chromatica Pro

High-performance CUDA kernels for 3D LUT application and color transforms.
Compiled at runtime using Numba CUDA.
"""

import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

try:
    from numba import cuda, float32
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False
    logger.warning("Numba CUDA not available")


if HAS_CUDA:
    
    @cuda.jit
    def apply_3d_lut_kernel(image, lut, output, lut_size):
        """
        CUDA kernel for 3D LUT application with trilinear interpolation.
        
        Args:
            image: Input image [H, W, 3]
            lut: 3D LUT [size, size, size, 3]
            output: Output image [H, W, 3]
            lut_size: LUT dimension
        """
        x, y = cuda.grid(2)
        
        h, w = image.shape[:2]
        
        if x < w and y < h:
            # Get RGB values
            r = image[y, x, 0]
            g = image[y, x, 1]
            b = image[y, x, 2]
            
            # Clamp to 0-1
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            b = max(0.0, min(1.0, b))
            
            # Scale to LUT coordinates
            scale = float32(lut_size - 1)
            r_scaled = r * scale
            g_scaled = g * scale
            b_scaled = b * scale
            
            # Integer coordinates
            r0 = int(r_scaled)
            g0 = int(g_scaled)
            b0 = int(b_scaled)
            
            r1 = min(r0 + 1, lut_size - 1)
            g1 = min(g0 + 1, lut_size - 1)
            b1 = min(b0 + 1, lut_size - 1)
            
            # Fractional parts
            dr = r_scaled - r0
            dg = g_scaled - g0
            db = b_scaled - b0
            
            # Trilinear interpolation for each output channel
            for c in range(3):
                c000 = lut[r0, g0, b0, c]
                c001 = lut[r0, g0, b1, c]
                c010 = lut[r0, g1, b0, c]
                c011 = lut[r0, g1, b1, c]
                c100 = lut[r1, g0, b0, c]
                c101 = lut[r1, g0, b1, c]
                c110 = lut[r1, g1, b0, c]
                c111 = lut[r1, g1, b1, c]
                
                # Interpolate
                c00 = c000 * (1 - dr) + c100 * dr
                c01 = c001 * (1 - dr) + c101 * dr
                c10 = c010 * (1 - dr) + c110 * dr
                c11 = c011 * (1 - dr) + c111 * dr
                
                c0 = c00 * (1 - dg) + c10 * dg
                c1 = c01 * (1 - dg) + c11 * dg
                
                output[y, x, c] = c0 * (1 - db) + c1 * db
    
    
    @cuda.jit
    def apply_matrix_3x4_kernel(image, matrix, output):
        """
        CUDA kernel for 3x4 matrix color transform.
        
        output = image @ matrix[:,:3].T + matrix[:,3]
        """
        x, y = cuda.grid(2)
        
        h, w = image.shape[:2]
        
        if x < w and y < h:
            r = image[y, x, 0]
            g = image[y, x, 1]
            b = image[y, x, 2]
            
            output[y, x, 0] = r * matrix[0, 0] + g * matrix[0, 1] + b * matrix[0, 2] + matrix[0, 3]
            output[y, x, 1] = r * matrix[1, 0] + g * matrix[1, 1] + b * matrix[1, 2] + matrix[1, 3]
            output[y, x, 2] = r * matrix[2, 0] + g * matrix[2, 1] + b * matrix[2, 2] + matrix[2, 3]
    
    
    @cuda.jit
    def apply_cdl_kernel(image, slope, offset, power, saturation, output):
        """
        CUDA kernel for ASC CDL transform.
        """
        x, y = cuda.grid(2)
        
        h, w = image.shape[:2]
        
        if x < w and y < h:
            r = image[y, x, 0]
            g = image[y, x, 1]
            b = image[y, x, 2]
            
            # Slope
            r = r * slope[0]
            g = g * slope[1]
            b = b * slope[2]
            
            # Offset
            r = r + offset[0]
            g = g + offset[1]
            b = b + offset[2]
            
            # Power (with sign preservation)
            if r >= 0:
                r = r ** power[0]
            else:
                r = -((-r) ** power[0])
                
            if g >= 0:
                g = g ** power[1]
            else:
                g = -((-g) ** power[1])
                
            if b >= 0:
                b = b ** power[2]
            else:
                b = -((-b) ** power[2])
            
            # Saturation
            if saturation != 1.0:
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                r = luma + saturation * (r - luma)
                g = luma + saturation * (g - luma)
                b = luma + saturation * (b - luma)
            
            output[y, x, 0] = r
            output[y, x, 1] = g
            output[y, x, 2] = b
    
    
    class CUDAProcessor:
        """
        CUDA-accelerated image processor.
        
        Example:
            >>> processor = CUDAProcessor()
            >>> result = processor.apply_lut(image, lut)
        """
        
        def __init__(self, block_size: int = 16):
            self.block_size = block_size
            logger.info("CUDA processor initialized")
        
        def apply_lut(self, image: npt.NDArray, lut: npt.NDArray) -> npt.NDArray:
            """Apply 3D LUT using CUDA."""
            h, w = image.shape[:2]
            lut_size = lut.shape[0]
            
            # Allocate device memory
            d_image = cuda.to_device(image.astype(np.float32))
            d_lut = cuda.to_device(lut.astype(np.float32))
            d_output = cuda.device_array((h, w, 3), dtype=np.float32)
            
            # Calculate grid
            threads_per_block = (self.block_size, self.block_size)
            blocks_per_grid_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # Launch kernel
            apply_3d_lut_kernel[blocks_per_grid, threads_per_block](
                d_image, d_lut, d_output, lut_size)
            
            # Copy back
            return d_output.copy_to_host()
        
        def apply_matrix(self, image: npt.NDArray, matrix: npt.NDArray) -> npt.NDArray:
            """Apply 3x4 matrix using CUDA."""
            h, w = image.shape[:2]
            
            d_image = cuda.to_device(image.astype(np.float32))
            d_matrix = cuda.to_device(matrix.astype(np.float32))
            d_output = cuda.device_array((h, w, 3), dtype=np.float32)
            
            threads_per_block = (self.block_size, self.block_size)
            blocks_per_grid_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            apply_matrix_3x4_kernel[blocks_per_grid, threads_per_block](
                d_image, d_matrix, d_output)
            
            return d_output.copy_to_host()
        
        def apply_cdl(self, image: npt.NDArray, 
                      slope: tuple, offset: tuple, power: tuple,
                      saturation: float) -> npt.NDArray:
            """Apply CDL using CUDA."""
            h, w = image.shape[:2]
            
            d_image = cuda.to_device(image.astype(np.float32))
            d_slope = cuda.to_device(np.array(slope, dtype=np.float32))
            d_offset = cuda.to_device(np.array(offset, dtype=np.float32))
            d_power = cuda.to_device(np.array(power, dtype=np.float32))
            d_output = cuda.device_array((h, w, 3), dtype=np.float32)
            
            threads_per_block = (self.block_size, self.block_size)
            blocks_per_grid_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            apply_cdl_kernel[blocks_per_grid, threads_per_block](
                d_image, d_slope, d_offset, d_power, saturation, d_output)
            
            return d_output.copy_to_host()

else:
    class CUDAProcessor:
        """Fallback when CUDA not available."""
        
        def __init__(self, block_size: int = 16):
            logger.warning("CUDA not available, using CPU fallback")
        
        def apply_lut(self, image, lut):
            from scipy.interpolate import RegularGridInterpolator
            size = lut.shape[0]
            x = np.linspace(0, 1, size)
            interp = RegularGridInterpolator((x, x, x), lut, method='linear')
            return interp(np.clip(image.reshape(-1, 3), 0, 1)).reshape(image.shape)
        
        def apply_matrix(self, image, matrix):
            flat = image.reshape(-1, 3)
            result = np.dot(flat, matrix[:, :3].T) + matrix[:, 3]
            return result.reshape(image.shape)
        
        def apply_cdl(self, image, slope, offset, power, saturation):
            result = image * np.array(slope) + np.array(offset)
            result = np.sign(result) * np.abs(result) ** np.array(power)
            if saturation != 1.0:
                luma = 0.2126*result[...,0] + 0.7152*result[...,1] + 0.0722*result[...,2]
                result = luma[..., np.newaxis] + saturation * (result - luma[..., np.newaxis])
            return result
