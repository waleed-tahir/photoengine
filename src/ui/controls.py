"""
CDL Controls and Curve Editors for Chromatica Pro

Professional controls for ASC CDL grading and curve editing.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Callable
from dataclasses import dataclass, field

# Optional Qt imports
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
        QGroupBox, QGridLayout, QPushButton, QTabWidget, QFrame
    )
    from PySide6.QtCore import Signal, Qt
    from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath
    HAS_QT = True
except ImportError:
    HAS_QT = False
    # Create dummy base classes
    class QWidget:
        pass
    class Signal:
        def __init__(self, *args): pass


@dataclass 
class CDLValues:
    """ASC CDL parameter values."""
    slope: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    power: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    saturation: float = 1.0


class CDLControls:
    """
    ASC CDL controls with color wheels and sliders.
    
    Provides professional grading controls for slope, offset, power, and saturation.
    Can be used standalone or integrated with Qt UI.
    """
    
    def __init__(self, on_change: Optional[Callable] = None):
        """
        Initialize CDL controls.
        
        Args:
            on_change: Callback when CDL values change
        """
        self.values = CDLValues()
        self._on_change = on_change
        self._callbacks = []
    
    def set_slope(self, r: float, g: float, b: float) -> None:
        """Set slope (gain) values."""
        self.values.slope = (
            max(0.0, min(4.0, r)),
            max(0.0, min(4.0, g)),
            max(0.0, min(4.0, b))
        )
        self._notify()
    
    def set_offset(self, r: float, g: float, b: float) -> None:
        """Set offset (lift) values."""
        self.values.offset = (
            max(-1.0, min(1.0, r)),
            max(-1.0, min(1.0, g)),
            max(-1.0, min(1.0, b))
        )
        self._notify()
    
    def set_power(self, r: float, g: float, b: float) -> None:
        """Set power (gamma) values."""
        self.values.power = (
            max(0.01, min(4.0, r)),
            max(0.01, min(4.0, g)),
            max(0.01, min(4.0, b))
        )
        self._notify()
    
    def set_saturation(self, sat: float) -> None:
        """Set saturation value."""
        self.values.saturation = max(0.0, min(4.0, sat))
        self._notify()
    
    def reset(self) -> None:
        """Reset to identity (no change)."""
        self.values = CDLValues()
        self._notify()
    
    def get_cdl_dict(self) -> Dict:
        """Get CDL values as dictionary."""
        return {
            'slope': self.values.slope,
            'offset': self.values.offset,
            'power': self.values.power,
            'saturation': self.values.saturation
        }
    
    def on_change(self, callback: Callable) -> None:
        """Register change callback."""
        self._callbacks.append(callback)
    
    def _notify(self) -> None:
        """Notify all callbacks of value change."""
        if self._on_change:
            self._on_change(self.values)
        for cb in self._callbacks:
            cb(self.values)


class CurveEditor:
    """
    Interactive curve editor for color grading.
    
    Provides a Bezier-based curve editor with control points
    for R, G, B, and Master luminance curves.
    """
    
    def __init__(self, num_points: int = 17, on_change: Optional[Callable] = None):
        """
        Initialize curve editor.
        
        Args:
            num_points: Number of points in the curve LUT
            on_change: Callback when curve changes
        """
        self.num_points = num_points
        self._on_change = on_change
        
        # Initialize identity curves (diagonal)
        self.curves = {
            'R': np.linspace(0, 1, num_points).copy(),
            'G': np.linspace(0, 1, num_points).copy(),
            'B': np.linspace(0, 1, num_points).copy(),
            'Master': np.linspace(0, 1, num_points).copy()
        }
        
        # Control points for editing (normalized 0-1)
        self.control_points = {
            'R': [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)],
            'G': [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)],
            'B': [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)],
            'Master': [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)]
        }
        
        self.active_channel = 'Master'
    
    def set_control_point(self, channel: str, index: int, x: float, y: float) -> None:
        """Set a control point position."""
        if channel not in self.control_points:
            raise ValueError(f"Invalid channel: {channel}")
        
        points = self.control_points[channel]
        if 0 <= index < len(points):
            # Keep x in order (can't cross other points)
            if index > 0:
                x = max(x, points[index-1][0] + 0.01)
            if index < len(points) - 1:
                x = min(x, points[index+1][0] - 0.01)
            
            # Keep first and last points at edges
            if index == 0:
                x = 0.0
            elif index == len(points) - 1:
                x = 1.0
            
            y = max(0.0, min(1.0, y))
            points[index] = (x, y)
            
            self._rebuild_curve(channel)
            self._notify()
    
    def add_control_point(self, channel: str, x: float, y: float) -> int:
        """Add a new control point."""
        points = self.control_points[channel]
        
        # Find insertion position
        insert_idx = 0
        for i, (px, _) in enumerate(points):
            if x > px:
                insert_idx = i + 1
        
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        points.insert(insert_idx, (x, y))
        
        self._rebuild_curve(channel)
        self._notify()
        return insert_idx
    
    def remove_control_point(self, channel: str, index: int) -> None:
        """Remove a control point (except endpoints)."""
        points = self.control_points[channel]
        if 0 < index < len(points) - 1:  # Don't remove endpoints
            points.pop(index)
            self._rebuild_curve(channel)
            self._notify()
    
    def reset_channel(self, channel: str) -> None:
        """Reset a channel to identity."""
        self.control_points[channel] = [
            (0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)
        ]
        self.curves[channel] = np.linspace(0, 1, self.num_points)
        self._notify()
    
    def reset_all(self) -> None:
        """Reset all channels to identity."""
        for channel in ['R', 'G', 'B', 'Master']:
            self.reset_channel(channel)
    
    def get_curve(self, channel: str) -> np.ndarray:
        """Get the curve LUT for a channel."""
        return self.curves[channel].copy()
    
    def get_all_curves(self) -> Dict[str, np.ndarray]:
        """Get all curve LUTs."""
        return {k: v.copy() for k, v in self.curves.items()}
    
    def _rebuild_curve(self, channel: str) -> None:
        """Rebuild curve LUT from control points using cubic interpolation."""
        points = self.control_points[channel]
        x_pts = np.array([p[0] for p in points])
        y_pts = np.array([p[1] for p in points])
        
        # Use monotonic interpolation for smooth curves
        from scipy.interpolate import PchipInterpolator
        
        x_out = np.linspace(0, 1, self.num_points)
        try:
            interp = PchipInterpolator(x_pts, y_pts)
            self.curves[channel] = np.clip(interp(x_out), 0, 1)
        except Exception:
            # Fallback to linear
            self.curves[channel] = np.interp(x_out, x_pts, y_pts)
    
    def on_change(self, callback: Callable) -> None:
        """Register change callback."""
        if self._on_change is None:
            self._on_change = callback
    
    def _notify(self) -> None:
        """Notify callback of curve change."""
        if self._on_change:
            self._on_change(self.curves)


class ControlPanel:
    """
    Combined control panel with CDL and curves.
    
    Provides a unified interface for color grading controls.
    """
    
    def __init__(self, on_change: Optional[Callable] = None):
        """
        Initialize control panel.
        
        Args:
            on_change: Callback when any control changes
        """
        self._on_change = on_change
        
        self.cdl = CDLControls(on_change=self._cdl_changed)
        self.curves = CurveEditor(on_change=self._curves_changed)
    
    def _cdl_changed(self, values: CDLValues) -> None:
        """Handle CDL change."""
        if self._on_change:
            self._on_change('cdl', values)
    
    def _curves_changed(self, curves: Dict) -> None:
        """Handle curves change."""
        if self._on_change:
            self._on_change('curves', curves)
    
    def get_parameters(self) -> Dict:
        """Get all parameters as a dictionary."""
        return {
            'cdl': self.cdl.get_cdl_dict(),
            'curves': self.curves.get_all_curves()
        }
    
    def reset(self) -> None:
        """Reset all controls to identity."""
        self.cdl.reset()
        self.curves.reset_all()


if HAS_QT:
    class CDLControlsWidget(QWidget):
        """Qt Widget for CDL controls."""
        
        valuesChanged = Signal(object)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._controls = CDLControls(on_change=self._emit_change)
            self._setup_ui()
        
        def _setup_ui(self):
            layout = QVBoxLayout(self)
            
            # Slope group
            slope_group = QGroupBox("Slope (Gain)")
            slope_layout = QGridLayout()
            
            for i, (channel, color) in enumerate([('R', 'red'), ('G', 'green'), ('B', 'blue')]):
                label = QLabel(channel)
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 400)
                slider.setValue(100)
                slider.valueChanged.connect(lambda v, c=i: self._slope_changed(c, v))
                slope_layout.addWidget(label, i, 0)
                slope_layout.addWidget(slider, i, 1)
            
            slope_group.setLayout(slope_layout)
            layout.addWidget(slope_group)
            
            # Saturation slider
            sat_layout = QHBoxLayout()
            sat_layout.addWidget(QLabel("Saturation"))
            sat_slider = QSlider(Qt.Horizontal)
            sat_slider.setRange(0, 400)
            sat_slider.setValue(100)
            sat_slider.valueChanged.connect(lambda v: self._controls.set_saturation(v / 100.0))
            sat_layout.addWidget(sat_slider)
            layout.addLayout(sat_layout)
            
            # Reset button
            reset_btn = QPushButton("Reset")
            reset_btn.clicked.connect(self._controls.reset)
            layout.addWidget(reset_btn)
        
        def _slope_changed(self, channel: int, value: int):
            slope = list(self._controls.values.slope)
            slope[channel] = value / 100.0
            self._controls.set_slope(*slope)
        
        def _emit_change(self, values):
            self.valuesChanged.emit(values)
        
        @property
        def controls(self):
            return self._controls


# Export all classes
__all__ = ['CDLControls', 'CDLValues', 'CurveEditor', 'ControlPanel']
if HAS_QT:
    __all__.append('CDLControlsWidget')
