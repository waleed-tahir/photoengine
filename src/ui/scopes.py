"""
Professional Video Scopes for Chromatica Pro UI

Waveform, Vectorscope, and Histogram Qt widgets.
"""

import numpy as np
from typing import Optional

# Optional Qt imports
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QComboBox, QGroupBox, QTabWidget, QFrame
    )
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QPainter, QPen, QColor, QImage, QPixmap
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:
        pass


class ScopeWidget:
    """
    Base class for video scopes.
    
    Provides common functionality for waveform, vectorscope, and histogram.
    Can be used standalone (returns numpy arrays) or with Qt.
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        self._buffer = np.zeros((height, width, 3), dtype=np.float32)
    
    def render(self, image: np.ndarray) -> np.ndarray:
        """
        Render scope from image.
        
        Args:
            image: Input image [H, W, 3] in 0-1 range
            
        Returns:
            Scope rendering [height, width, 3]
        """
        raise NotImplementedError
    
    def _clear_buffer(self):
        """Clear the render buffer."""
        self._buffer.fill(0)


class WaveformWidget(ScopeWidget):
    """
    Waveform monitor scope.
    
    Displays luminance or RGB parade waveform.
    """
    
    def __init__(self, width: int = 360, height: int = 240, mode: str = 'luma'):
        """
        Initialize waveform widget.
        
        Args:
            width: Scope width
            height: Scope height  
            mode: 'luma', 'rgb_parade', or 'rgb_overlay'
        """
        super().__init__(width, height)
        self.mode = mode
    
    def render(self, image: np.ndarray) -> np.ndarray:
        """Render waveform from image."""
        self._clear_buffer()
        
        if self.mode == 'luma':
            return self._render_luma(image)
        elif self.mode == 'rgb_parade':
            return self._render_rgb_parade(image)
        else:
            return self._render_luma(image)
    
    def _render_luma(self, image: np.ndarray) -> np.ndarray:
        """Render luminance waveform."""
        h, w = image.shape[:2]
        
        # Calculate luminance
        luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        
        # Map columns
        x_scale = self.width / w
        
        for col in range(w):
            x = int(col * x_scale)
            if x >= self.width:
                x = self.width - 1
            
            column_values = luma[:, col]
            for val in column_values:
                y = int((1 - np.clip(val, 0, 1)) * (self.height - 1))
                if 0 <= y < self.height:
                    self._buffer[y, x] += 0.02
        
        # Normalize and apply color
        self._buffer = np.clip(self._buffer, 0, 1)
        
        # Make it green like traditional scopes
        result = np.zeros_like(self._buffer)
        result[..., 1] = self._buffer[..., 0]
        
        return result
    
    def _render_rgb_parade(self, image: np.ndarray) -> np.ndarray:
        """Render RGB parade waveform."""
        h, w = image.shape[:2]
        
        third_width = self.width // 3
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB
        
        for ch in range(3):
            channel = image[..., ch]
            x_offset = ch * third_width
            x_scale = third_width / w
            
            for col in range(w):
                x = int(col * x_scale) + x_offset
                if x >= self.width:
                    x = self.width - 1
                
                column_values = channel[:, col]
                for val in column_values:
                    y = int((1 - np.clip(val, 0, 1)) * (self.height - 1))
                    if 0 <= y < self.height:
                        self._buffer[y, x, ch] += 0.02
        
        return np.clip(self._buffer, 0, 1)


class VectorscopeWidget(ScopeWidget):
    """
    Vectorscope for chrominance visualization.
    
    Displays color distribution in IQ/UV space with standard color targets.
    """
    
    def __init__(self, size: int = 256):
        super().__init__(size, size)
        self.size = size
        self._graticule = self._create_graticule()
    
    def _create_graticule(self) -> np.ndarray:
        """Create vectorscope graticule with color targets."""
        graticule = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        center = self.size // 2
        
        # Draw circles
        for radius_pct in [0.25, 0.5, 0.75, 1.0]:
            radius = int(center * radius_pct)
            for angle in np.linspace(0, 2 * np.pi, 360):
                x = int(center + radius * np.cos(angle))
                y = int(center + radius * np.sin(angle))
                if 0 <= x < self.size and 0 <= y < self.size:
                    graticule[y, x] = 0.2
        
        # Color target boxes (approx positions for Rec.709)
        targets = [
            (90, 0.45, (1, 0, 0)),    # Red at ~90°
            (60, 0.32, (1, 1, 0)),    # Yellow
            (120, 0.28, (0, 1, 0)),   # Green  
            (180, 0.45, (0, 1, 1)),   # Cyan
            (240, 0.32, (0, 0, 1)),   # Blue
            (300, 0.28, (1, 0, 1)),   # Magenta
        ]
        
        for angle_deg, sat, color in targets:
            angle = np.radians(angle_deg)
            radius = int(center * sat * 0.8)
            x = int(center + radius * np.cos(angle))
            y = int(center - radius * np.sin(angle))
            
            # Draw small box
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if abs(dx) == 4 or abs(dy) == 4:
                        px, py = x + dx, y + dy
                        if 0 <= px < self.size and 0 <= py < self.size:
                            graticule[py, px] = color
        
        return graticule
    
    def render(self, image: np.ndarray) -> np.ndarray:
        """Render vectorscope from image."""
        result = self._graticule.copy()
        
        h, w = image.shape[:2]
        center = self.size // 2
        
        # Convert RGB to chrominance (simplified I/Q)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        
        # YIQ transform
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b  # In-phase
        q = 0.211 * r - 0.523 * g + 0.312 * b  # Quadrature
        
        # Plot points
        scale = center * 0.9
        i_flat = i.flatten()
        q_flat = q.flatten()
        
        # Subsample for performance
        step = max(1, len(i_flat) // 50000)
        
        for idx in range(0, len(i_flat), step):
            x = int(center + i_flat[idx] * scale)
            y_coord = int(center - q_flat[idx] * scale)
            
            if 0 <= x < self.size and 0 <= y_coord < self.size:
                result[y_coord, x] += 0.1
        
        return np.clip(result, 0, 1)


class HistogramWidget(ScopeWidget):
    """
    RGB Histogram display.
    """
    
    def __init__(self, width: int = 256, height: int = 100):
        super().__init__(width, height)
        self.bins = width
    
    def render(self, image: np.ndarray) -> np.ndarray:
        """Render RGB histogram."""
        self._clear_buffer()
        
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        
        for ch in range(3):
            channel = image[..., ch].flatten()
            hist, _ = np.histogram(channel, bins=self.bins, range=(0, 1))
            
            # Normalize
            if hist.max() > 0:
                hist = hist / hist.max()
            
            # Draw histogram
            for x, val in enumerate(hist):
                h = int(val * (self.height - 1))
                for y in range(self.height - h, self.height):
                    self._buffer[y, x, ch] = max(self._buffer[y, x, ch], 0.8)
        
        return self._buffer


class ScopesPanel:
    """
    Combined scopes panel with waveform, vectorscope, and histogram.
    """
    
    def __init__(self, width: int = 320, height: int = 240):
        self.waveform = WaveformWidget(width, height, mode='luma')
        self.vectorscope = VectorscopeWidget(min(width, height))
        self.histogram = HistogramWidget(width, height // 2)
    
    def render_all(self, image: np.ndarray) -> dict:
        """Render all scopes and return as dictionary."""
        return {
            'waveform': self.waveform.render(image),
            'vectorscope': self.vectorscope.render(image),
            'histogram': self.histogram.render(image)
        }


if HAS_QT:
    class QScopeWidget(QWidget):
        """Qt wrapper for scope widgets."""
        
        def __init__(self, scope: ScopeWidget, parent=None):
            super().__init__(parent)
            self._scope = scope
            self._pixmap = None
            self.setMinimumSize(scope.width, scope.height)
        
        def update_image(self, image: np.ndarray):
            """Update scope with new image."""
            rendered = self._scope.render(image)
            
            # Convert to QImage
            rendered_uint8 = (np.clip(rendered, 0, 1) * 255).astype(np.uint8)
            h, w, ch = rendered_uint8.shape
            bytes_per_line = ch * w
            
            qimage = QImage(rendered_uint8.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimage.copy())
            self.update()
        
        def paintEvent(self, event):
            if self._pixmap:
                painter = QPainter(self)
                painter.drawPixmap(0, 0, self._pixmap)


# Export all
__all__ = ['ScopeWidget', 'WaveformWidget', 'VectorscopeWidget', 
           'HistogramWidget', 'ScopesPanel']
if HAS_QT:
    __all__.append('QScopeWidget')
