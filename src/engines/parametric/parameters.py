"""
Parametric Parameters for Color Matching

Defines the 160-parameter structure for professional color matching:
- 3x4 matrix (12 params)
- RGB curves (15 params) + master curve (5 params)  
- Hue vs Sat/Hue 2D tables (128 params)
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import json


@dataclass
class CDLParameters:
    """ASC CDL (Color Decision List) parameters"""
    slope: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    power: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    saturation: float = 1.0
    
    def to_vector(self) -> npt.NDArray[np.float64]:
        return np.array([*self.slope, *self.offset, *self.power, self.saturation])
    
    @classmethod
    def from_vector(cls, vec: npt.NDArray) -> "CDLParameters":
        return cls(
            slope=tuple(vec[0:3]),
            offset=tuple(vec[3:6]),
            power=tuple(vec[6:9]),
            saturation=float(vec[9])
        )
    
    def to_dict(self) -> Dict:
        return {
            'slope': list(self.slope),
            'offset': list(self.offset),
            'power': list(self.power),
            'saturation': self.saturation
        }


@dataclass
class ParametricParameters:
    """
    Complete parametric color matching parameters.
    
    Total: 160 parameters for professional matching:
    - matrix: 3x4 color matrix (12 params)
    - rgb_curves: 5 points each for R, G, B (15 params)
    - master_curve: 5 points (5 params)
    - hue_vs_sat: 8x8 2D table (64 params)
    - hue_vs_hue: 8x8 2D table (64 params)
    """
    matrix: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(3, 4))
    rgb_curves: Dict[str, npt.NDArray[np.float64]] = field(default_factory=lambda: {
        'R': np.linspace(0, 1, 5),
        'G': np.linspace(0, 1, 5),
        'B': np.linspace(0, 1, 5)
    })
    master_curve: npt.NDArray[np.float64] = field(default_factory=lambda: np.linspace(0, 1, 5))
    hue_vs_sat: npt.NDArray[np.float64] = field(default_factory=lambda: np.ones((8, 8)))
    hue_vs_hue: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((8, 8)))
    
    # Parameter indices for vector conversion
    PARAM_DEFS = {
        'matrix': (0, 12),
        'rgb_curves': (12, 27),
        'master_curve': (27, 32),
        'hue_vs_sat': (32, 96),
        'hue_vs_hue': (96, 160)
    }
    TOTAL_PARAMS = 160
    
    def to_vector(self) -> npt.NDArray[np.float64]:
        """Convert parameters to optimization vector."""
        vec = np.zeros(self.TOTAL_PARAMS, dtype=np.float64)
        
        # Matrix (3x4 flattened)
        vec[0:12] = self.matrix.flatten()
        
        # RGB curves
        vec[12:17] = self.rgb_curves['R']
        vec[17:22] = self.rgb_curves['G']
        vec[22:27] = self.rgb_curves['B']
        
        # Master curve
        vec[27:32] = self.master_curve
        
        # 2D tables
        vec[32:96] = self.hue_vs_sat.flatten()
        vec[96:160] = self.hue_vs_hue.flatten()
        
        return vec
    
    @classmethod
    def from_vector(cls, vec: npt.NDArray) -> "ParametricParameters":
        """Create parameters from optimization vector."""
        return cls(
            matrix=vec[0:12].reshape(3, 4),
            rgb_curves={
                'R': vec[12:17].copy(),
                'G': vec[17:22].copy(),
                'B': vec[22:27].copy()
            },
            master_curve=vec[27:32].copy(),
            hue_vs_sat=vec[32:96].reshape(8, 8),
            hue_vs_hue=vec[96:160].reshape(8, 8)
        )
    
    @classmethod
    def identity(cls) -> "ParametricParameters":
        """Create identity (no-op) parameters."""
        return cls()
    
    @classmethod
    def from_cdl(cls, cdl: CDLParameters) -> "ParametricParameters":
        """Create parameters from CDL values."""
        params = cls.identity()
        
        # Convert CDL to matrix (diagonal with offset)
        params.matrix = np.array([
            [cdl.slope[0], 0, 0, cdl.offset[0]],
            [0, cdl.slope[1], 0, cdl.offset[1]],
            [0, 0, cdl.slope[2], cdl.offset[2]]
        ], dtype=np.float64)
        
        # Power goes into curves (gamma curves)
        for i, (ch, power) in enumerate(zip(['R', 'G', 'B'], cdl.power)):
            x = np.linspace(0, 1, 5)
            params.rgb_curves[ch] = np.power(x, 1.0 / power)
        
        return params
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'matrix': self.matrix.tolist(),
            'rgb_curves': {k: v.tolist() for k, v in self.rgb_curves.items()},
            'master_curve': self.master_curve.tolist(),
            'hue_vs_sat': self.hue_vs_sat.tolist(),
            'hue_vs_hue': self.hue_vs_hue.tolist()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ParametricParameters":
        """Create from dict."""
        return cls(
            matrix=np.array(d['matrix']),
            rgb_curves={k: np.array(v) for k, v in d['rgb_curves'].items()},
            master_curve=np.array(d['master_curve']),
            hue_vs_sat=np.array(d['hue_vs_sat']),
            hue_vs_hue=np.array(d['hue_vs_hue'])
        )
    
    def save(self, path: str) -> None:
        """Save parameters to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ParametricParameters":
        """Load parameters from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def get_parameter_bounds() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get lower and upper bounds for parameter optimization."""
    lower = np.zeros(ParametricParameters.TOTAL_PARAMS)
    upper = np.ones(ParametricParameters.TOTAL_PARAMS)
    
    # Matrix bounds (3x4 flattened: indices 0-11)
    # Allow full range for all matrix elements to include identity (1.0 on diagonal)
    lower[0:12] = -2.0
    upper[0:12] = 2.0
    
    # Curves: 0-1 (already set)
    
    # Hue vs Sat: 0.5x to 2.0x multiplier
    lower[32:96] = 0.5
    upper[32:96] = 2.0
    
    # Hue vs Hue: -0.5 to +0.5 radians shift
    lower[96:160] = -0.5
    upper[96:160] = 0.5
    
    return lower, upper

