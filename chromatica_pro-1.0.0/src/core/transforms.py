"""
Color Transformation Primitives for Chromatica Pro

Provides: Matrix transforms, CDL operations, Curve interpolation, 1D/2D/3D LUT interpolation
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from scipy.interpolate import interp1d, RegularGridInterpolator, PchipInterpolator
import logging

logger = logging.getLogger(__name__)


@dataclass
class Curve:
    """Curve definition with control points"""
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    interpolation: str = "cubic"
    
    def __post_init__(self):
        if self.interpolation == "linear":
            self._interp = interp1d(self.x, self.y, kind='linear', 
                                    bounds_error=False, fill_value=(self.y[0], self.y[-1]))
        elif self.interpolation == "cubic":
            self._interp = interp1d(self.x, self.y, kind='cubic',
                                    bounds_error=False, fill_value=(self.y[0], self.y[-1]))
        elif self.interpolation == "monotonic":
            self._interp = PchipInterpolator(self.x, self.y, extrapolate=True)
    
    def __call__(self, values: npt.NDArray) -> npt.NDArray:
        return self._interp(values)


@dataclass 
class LUT1D:
    """1D Look-Up Table"""
    values: npt.NDArray[np.float64]
    size: int = 0
    domain: Tuple[float, float] = (0.0, 1.0)
    
    def __post_init__(self):
        if self.size == 0:
            self.size = len(self.values)
    
    def apply(self, image: npt.NDArray) -> npt.NDArray:
        min_val, max_val = self.domain
        normalized = (image - min_val) / (max_val - min_val)
        indices = normalized * (self.size - 1)
        idx_low = np.clip(np.floor(indices).astype(np.int32), 0, self.size - 1)
        idx_high = np.clip(np.ceil(indices).astype(np.int32), 0, self.size - 1)
        frac = indices - idx_low
        return self.values[idx_low] * (1 - frac) + self.values[idx_high] * frac


@dataclass
class LUT3D:
    """3D Look-Up Table for color transformations"""
    values: npt.NDArray[np.float64]  # [size, size, size, 3]
    size: int = 0
    domain: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    
    def __post_init__(self):
        if self.size == 0:
            self.size = self.values.shape[0]
        x = np.linspace(self.domain[0][0], self.domain[0][1], self.size)
        y = np.linspace(self.domain[1][0], self.domain[1][1], self.size)
        z = np.linspace(self.domain[2][0], self.domain[2][1], self.size)
        self._interp = RegularGridInterpolator((x, y, z), self.values,
                                                method='linear', bounds_error=False)
    
    def apply(self, image: npt.NDArray) -> npt.NDArray:
        original_shape = image.shape
        points = np.clip(image.reshape(-1, 3), 
                        [d[0] for d in self.domain], [d[1] for d in self.domain])
        return self._interp(points).reshape(original_shape)
    
    def apply_tetrahedral(self, image: npt.NDArray) -> npt.NDArray:
        """Apply 3D LUT using tetrahedral interpolation (industry standard)."""
        image = np.asarray(image, dtype=np.float64)
        original_shape = image.shape
        rgb = image.reshape(-1, 3)
        size, lut = self.size, self.values
        
        r = np.clip(rgb[:, 0], 0, 1) * (size - 1)
        g = np.clip(rgb[:, 1], 0, 1) * (size - 1)
        b = np.clip(rgb[:, 2], 0, 1) * (size - 1)
        
        r0, g0, b0 = np.floor(r).astype(np.int32), np.floor(g).astype(np.int32), np.floor(b).astype(np.int32)
        r1, g1, b1 = np.minimum(r0+1, size-1), np.minimum(g0+1, size-1), np.minimum(b0+1, size-1)
        dr, dg, db = r - r0, g - g0, b - b0
        
        result = np.zeros((len(rgb), 3), dtype=np.float64)
        
        # 6 tetrahedra cases
        for mask, corners in [
            ((dr >= dg) & (dg >= db), [(r0,g0,b0),(r1,g0,b0),(r1,g1,b0),(r1,g1,b1)]),
            ((dr >= db) & (db >= dg), [(r0,g0,b0),(r1,g0,b0),(r1,g0,b1),(r1,g1,b1)]),
            ((dg >= dr) & (dr >= db), [(r0,g0,b0),(r0,g1,b0),(r1,g1,b0),(r1,g1,b1)]),
            ((dg >= db) & (db >= dr), [(r0,g0,b0),(r0,g1,b0),(r0,g1,b1),(r1,g1,b1)]),
            ((db >= dr) & (dr >= dg), [(r0,g0,b0),(r0,g0,b1),(r1,g0,b1),(r1,g1,b1)]),
            ((db >= dg) & (dg >= dr), [(r0,g0,b0),(r0,g0,b1),(r0,g1,b1),(r1,g1,b1)]),
        ]:
            if np.any(mask):
                c = [lut[corners[i][0][mask], corners[i][1][mask], corners[i][2][mask]] for i in range(4)]
                d = [dr[mask, np.newaxis], dg[mask, np.newaxis], db[mask, np.newaxis]]
                result[mask] = c[0] + (c[1]-c[0])*d[0] + (c[2]-c[1])*d[1] + (c[3]-c[2])*d[2]
        
        return result.reshape(original_shape)


class ColorTransforms:
    """Collection of color transformation utilities."""
    
    @staticmethod
    def apply_matrix_3x3(image: npt.NDArray, matrix: npt.NDArray) -> npt.NDArray[np.float64]:
        image, matrix = np.asarray(image, dtype=np.float64), np.asarray(matrix, dtype=np.float64)
        return np.dot(image.reshape(-1, 3), matrix.T).reshape(image.shape)
    
    @staticmethod
    def apply_matrix_3x4(image: npt.NDArray, matrix: npt.NDArray) -> npt.NDArray[np.float64]:
        image, matrix = np.asarray(image, dtype=np.float64), np.asarray(matrix, dtype=np.float64)
        return (np.dot(image.reshape(-1, 3), matrix[:,:3].T) + matrix[:,3]).reshape(image.shape)
    
    @staticmethod
    def apply_cdl(image: npt.NDArray,
                  slope: Tuple[float,float,float] = (1.0,1.0,1.0),
                  offset: Tuple[float,float,float] = (0.0,0.0,0.0),
                  power: Tuple[float,float,float] = (1.0,1.0,1.0),
                  saturation: float = 1.0) -> npt.NDArray[np.float64]:
        """Apply ASC CDL transform: out = (in * slope + offset)^power"""
        image = np.asarray(image, dtype=np.float64)
        result = image * np.array(slope) + np.array(offset)
        result = np.sign(result) * np.abs(result) ** np.array(power)
        if saturation != 1.0:
            luma = 0.2126*result[...,0] + 0.7152*result[...,1] + 0.0722*result[...,2]
            result = luma[...,np.newaxis] + saturation * (result - luma[...,np.newaxis])
        return result
    
    @staticmethod
    def apply_lift_gamma_gain(image: npt.NDArray,
                              lift: Tuple[float,float,float] = (0.0,0.0,0.0),
                              gamma: Tuple[float,float,float] = (1.0,1.0,1.0),
                              gain: Tuple[float,float,float] = (1.0,1.0,1.0)) -> npt.NDArray[np.float64]:
        """Apply Lift/Gamma/Gain: out = (gain * (in + lift * (1 - in)))^(1/gamma)"""
        image = np.asarray(image, dtype=np.float64)
        result = image + np.array(lift) * (1.0 - image)
        result = result * np.array(gain)
        return np.sign(result) * np.abs(result) ** (1.0 / np.array(gamma))
    
    @staticmethod
    def apply_curve(image: npt.NDArray, curve: Union[Curve, npt.NDArray],
                    channel: Optional[int] = None) -> npt.NDArray[np.float64]:
        image = np.asarray(image, dtype=np.float64)
        if isinstance(curve, np.ndarray):
            curve = Curve(np.linspace(0, 1, len(curve)), curve, interpolation="monotonic")
        result = image.copy()
        channels = [channel] if channel is not None else range(3)
        for c in channels:
            result[..., c] = curve(result[..., c])
        return result
    
    @staticmethod
    def apply_contrast(image: npt.NDArray, contrast: float, pivot: float = 0.18) -> npt.NDArray[np.float64]:
        return pivot + (np.asarray(image, dtype=np.float64) - pivot) * contrast
    
    @staticmethod
    def apply_exposure(image: npt.NDArray, stops: float) -> npt.NDArray[np.float64]:
        return np.asarray(image, dtype=np.float64) * (2.0 ** stops)
    
    @staticmethod
    def apply_saturation(image: npt.NDArray, saturation: float) -> npt.NDArray[np.float64]:
        image = np.asarray(image, dtype=np.float64)
        luma = 0.2126*image[...,0] + 0.7152*image[...,1] + 0.0722*image[...,2]
        return luma[...,np.newaxis] + saturation * (image - luma[...,np.newaxis])
    
    @staticmethod
    def apply_white_balance(image: npt.NDArray, temperature: float, tint: float) -> npt.NDArray[np.float64]:
        result = np.asarray(image, dtype=np.float64).copy()
        if temperature != 0:
            result[..., 0] *= (1 + temperature * 0.5)
            result[..., 2] *= (1 - temperature * 0.5)
        if tint != 0:
            result[..., 1] *= (1 - tint * 0.3)
        return result
    
    @staticmethod
    def create_identity_3d_lut(size: int = 33) -> LUT3D:
        x = np.linspace(0, 1, size)
        r, g, b = np.meshgrid(x, x, x, indexing='ij')
        return LUT3D(values=np.stack([r, g, b], axis=-1), size=size)
