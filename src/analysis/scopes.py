"""
Professional Video Scopes

Waveform, Vectorscope, and Histogram for color analysis.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class WaveformScope:
    """
    Professional waveform monitor for luminance/RGB analysis.
    
    Renders luma or RGB parade waveform common in video production.
    """
    
    def __init__(self, width: int = 720, height: int = 480):
        self.width = width
        self.height = height
    
    def render_luma(self, image: npt.NDArray, 
                    scale: float = 1.0) -> npt.NDArray:
        """
        Render luminance waveform.
        
        Args:
            image: Input RGB image [H, W, 3]
            scale: Intensity scale
            
        Returns:
            Waveform image [height, width, 3]
        """
        # Calculate luminance
        luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        
        # Create output
        waveform = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        h, w = luma.shape
        
        # Subsample columns
        for x_out in range(self.width):
            x_in = int(x_out * w / self.width)
            
            # Get column values
            col = luma[:, x_in]
            
            # Map to waveform Y coordinates (inverted: 0 at bottom, 1 at top)
            y_coords = ((1.0 - np.clip(col, 0, 1)) * (self.height - 1)).astype(np.int32)
            
            # Accumulate
            for y in y_coords:
                if 0 <= y < self.height:
                    waveform[y, x_out] += scale * 0.02
        
        # Normalize and colorize (green)
        waveform = np.clip(waveform, 0, 1)
        waveform[..., 0] *= 0.3  # Less red
        waveform[..., 2] *= 0.3  # Less blue
        
        # Add graticule lines at 0%, 50%, 100%
        for level in [0.0, 0.5, 1.0]:
            y = int((1.0 - level) * (self.height - 1))
            waveform[y, :] = np.maximum(waveform[y, :], [0.2, 0.2, 0.2])
        
        return waveform
    
    def render_rgb_parade(self, image: npt.NDArray,
                          scale: float = 1.0) -> npt.NDArray:
        """
        Render RGB parade waveform.
        
        Args:
            image: Input RGB image [H, W, 3]
            scale: Intensity scale
            
        Returns:
            RGB parade waveform [height, width, 3]
        """
        parade = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        section_width = self.width // 3
        h, w = image.shape[:2]
        
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB
        
        for c, (r, g, b) in enumerate(colors):
            channel = image[..., c]
            x_offset = c * section_width
            
            for x_out in range(section_width):
                x_in = int(x_out * w / section_width)
                col = channel[:, x_in]
                y_coords = ((1.0 - np.clip(col, 0, 1)) * (self.height - 1)).astype(np.int32)
                
                for y in y_coords:
                    if 0 <= y < self.height:
                        parade[y, x_offset + x_out, 0] += r * scale * 0.02
                        parade[y, x_offset + x_out, 1] += g * scale * 0.02
                        parade[y, x_offset + x_out, 2] += b * scale * 0.02
        
        return np.clip(parade, 0, 1)


class VectorscopeRenderer:
    """
    Professional vectorscope for chrominance analysis.
    
    Displays color distribution in 2D Cb/Cr space.
    """
    
    def __init__(self, size: int = 512):
        self.size = size
        self.center = size // 2
        self.radius = size // 2 - 20
        
        # Pre-render background
        self._background = self._create_background()
    
    def _create_background(self) -> npt.NDArray:
        """Create vectorscope background with color targets."""
        bg = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Draw graticule circles
        y, x = np.ogrid[:self.size, :self.size]
        cx, cy = self.center, self.center
        
        for r_frac in [0.25, 0.5, 0.75, 1.0]:
            r = int(self.radius * r_frac)
            mask = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r) < 1
            bg[mask] = [0.15, 0.15, 0.15]
        
        # Draw crosshairs
        bg[cy-1:cy+2, :] = np.maximum(bg[cy-1:cy+2, :], [0.1, 0.1, 0.1])
        bg[:, cx-1:cx+2] = np.maximum(bg[:, cx-1:cx+2], [0.1, 0.1, 0.1])
        
        # Color target boxes (SMPTE colors)
        targets = [
            (0.615, 0.515, (1, 0, 0), "R"),    # Red
            (0.385, 0.485, (0, 1, 1), "Cy"),   # Cyan
            (0.530, 0.165, (0, 1, 0), "G"),    # Green
            (0.470, 0.835, (1, 0, 1), "Mg"),   # Magenta
            (0.445, 0.385, (0, 0, 1), "B"),    # Blue
            (0.555, 0.615, (1, 1, 0), "Y"),    # Yellow
        ]
        
        for cb, cr, color, _ in targets:
            tx = int(self.center + (cb - 0.5) * 2 * self.radius)
            ty = int(self.center - (cr - 0.5) * 2 * self.radius)
            if 5 < tx < self.size - 5 and 5 < ty < self.size - 5:
                bg[ty-3:ty+4, tx-3:tx+4] = color
        
        return bg
    
    def render(self, image: npt.NDArray, scale: float = 1.0) -> npt.NDArray:
        """
        Render vectorscope.
        
        Args:
            image: Input RGB image [H, W, 3] in 0-1 range
            scale: Intensity scale
            
        Returns:
            Vectorscope image [size, size, 3]
        """
        scope = self._background.copy()
        
        # Convert RGB to YCbCr
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        
        # Flatten and subsample
        cb_flat = cb.flatten()[::10]
        cr_flat = cr.flatten()[::10]
        
        # Map to scope coordinates
        x_coords = ((cb_flat - 0.5) * 2 * self.radius + self.center).astype(np.int32)
        y_coords = (self.center - (cr_flat - 0.5) * 2 * self.radius).astype(np.int32)
        
        # Valid points
        valid = (x_coords >= 0) & (x_coords < self.size) & \
                (y_coords >= 0) & (y_coords < self.size)
        
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        
        # Accumulate with original color
        r_flat = image[..., 0].flatten()[::10][valid]
        g_flat = image[..., 1].flatten()[::10][valid]
        b_flat = image[..., 2].flatten()[::10][valid]
        
        for i in range(len(x_coords)):
            scope[y_coords[i], x_coords[i], 0] += r_flat[i] * scale * 0.01
            scope[y_coords[i], x_coords[i], 1] += g_flat[i] * scale * 0.01
            scope[y_coords[i], x_coords[i], 2] += b_flat[i] * scale * 0.01
        
        return np.clip(scope, 0, 1)


class HistogramRenderer:
    """Render RGB histogram."""
    
    def __init__(self, width: int = 256, height: int = 150):
        self.width = width
        self.height = height
    
    def render(self, image: npt.NDArray, bins: int = 256) -> npt.NDArray:
        """
        Render RGB histogram.
        
        Args:
            image: Input RGB image [H, W, 3]
            bins: Number of histogram bins
            
        Returns:
            Histogram image [height, width, 3]
        """
        histogram = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        colors = [(1, 0.3, 0.3), (0.3, 1, 0.3), (0.3, 0.3, 1)]
        
        for c, color in enumerate(colors):
            # Calculate histogram
            hist, _ = np.histogram(image[..., c].flatten(), bins=bins, range=(0, 1))
            
            # Normalize
            hist = hist.astype(np.float32)
            if hist.max() > 0:
                hist = hist / hist.max()
            
            # Draw
            for x in range(min(bins, self.width)):
                h = int(hist[x] * (self.height - 1))
                if h > 0:
                    histogram[-h:, x, 0] = max(histogram[-h:, x, 0].max(), color[0] * 0.7)
                    histogram[-h:, x, 1] = max(histogram[-h:, x, 1].max(), color[1] * 0.7)
                    histogram[-h:, x, 2] = max(histogram[-h:, x, 2].max(), color[2] * 0.7)
        
        return histogram
