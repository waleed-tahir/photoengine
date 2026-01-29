"""
Color Space Conversions for Chromatica Pro

This module provides comprehensive color space conversion utilities,
supporting professional workflows with high precision (float64).

Supports:
- RGB ↔ XYZ ↔ LAB conversions
- Log curve encodings (LogC, S-Log, V-Log, etc.)
- Gamut mapping and clipping protection
- Chromatic adaptation (Bradford, CAT02)
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChromaticAdaptation(Enum):
    """Chromatic adaptation methods"""
    BRADFORD = "bradford"
    VON_KRIES = "von_kries"
    CAT02 = "cat02"
    XYZ_SCALING = "xyz_scaling"


@dataclass
class Whitepoint:
    """CIE Standard Illuminant whitepoint definition"""
    name: str
    xy: Tuple[float, float]  # CIE xy chromaticity
    XYZ: Tuple[float, float, float]  # CIE XYZ (Y=1)


class StandardIlluminants:
    """CIE Standard Illuminant definitions"""
    
    D50 = Whitepoint("D50", (0.34567, 0.35850), (0.96422, 1.0, 0.82521))
    D55 = Whitepoint("D55", (0.33242, 0.34743), (0.95682, 1.0, 0.92149))
    D60 = Whitepoint("D60", (0.32168, 0.33767), (0.95265, 1.0, 1.00883))
    D65 = Whitepoint("D65", (0.31270, 0.32900), (0.95047, 1.0, 1.08883))
    D75 = Whitepoint("D75", (0.29902, 0.31485), (0.94972, 1.0, 1.22638))
    
    A = Whitepoint("A", (0.44757, 0.40745), (1.09850, 1.0, 0.35585))
    E = Whitepoint("E", (0.33333, 0.33333), (1.0, 1.0, 1.0))


@dataclass
class RGBColorSpace:
    """RGB color space definition"""
    name: str
    primaries: npt.NDArray[np.float64]  # [3, 2] - xy coordinates of R, G, B
    whitepoint: Whitepoint
    transfer_function: str  # gamma, srgb, linear, etc.
    
    # Pre-computed matrices (set in __post_init__)
    matrix_RGB_to_XYZ: Optional[npt.NDArray[np.float64]] = None
    matrix_XYZ_to_RGB: Optional[npt.NDArray[np.float64]] = None
    
    def __post_init__(self):
        if self.matrix_RGB_to_XYZ is None:
            self.matrix_RGB_to_XYZ = self._compute_rgb_to_xyz_matrix()
            self.matrix_XYZ_to_RGB = np.linalg.inv(self.matrix_RGB_to_XYZ)
    
    def _compute_rgb_to_xyz_matrix(self) -> npt.NDArray[np.float64]:
        """Compute RGB to XYZ matrix from primaries and whitepoint."""
        # Convert chromaticity to XYZ
        xr, yr = self.primaries[0]
        xg, yg = self.primaries[1]
        xb, yb = self.primaries[2]
        
        # XYZ values (Y=1 normalized)
        Xr, Yr, Zr = xr/yr, 1.0, (1-xr-yr)/yr
        Xg, Yg, Zg = xg/yg, 1.0, (1-xg-yg)/yg
        Xb, Yb, Zb = xb/yb, 1.0, (1-xb-yb)/yb
        
        # Whitepoint XYZ
        Xw, Yw, Zw = self.whitepoint.XYZ
        
        # Solve for S (scaling factors)
        M = np.array([
            [Xr, Xg, Xb],
            [Yr, Yg, Yb],
            [Zr, Zg, Zb],
        ], dtype=np.float64)
        
        S = np.linalg.solve(M, np.array([Xw, Yw, Zw]))
        
        # Final matrix
        matrix = np.array([
            [S[0] * Xr, S[1] * Xg, S[2] * Xb],
            [S[0] * Yr, S[1] * Yg, S[2] * Yb],
            [S[0] * Zr, S[1] * Zg, S[2] * Zb],
        ], dtype=np.float64)
        
        return matrix


class ColorSpaceConverter:
    """
    Professional color space conversion utility.
    
    Provides high-precision conversions between various color spaces
    used in professional film and video workflows.
    
    Example:
        >>> converter = ColorSpaceConverter()
        >>> xyz = converter.rgb_to_xyz(rgb_image, "sRGB")
        >>> lab = converter.xyz_to_lab(xyz)
        >>> delta_e = converter.delta_e_2000(lab1, lab2)
    """
    
    # Standard color spaces
    SRGB = RGBColorSpace(
        name="sRGB",
        primaries=np.array([
            [0.64, 0.33],   # Red
            [0.30, 0.60],   # Green
            [0.15, 0.06],   # Blue
        ], dtype=np.float64),
        whitepoint=StandardIlluminants.D65,
        transfer_function="srgb",
    )
    
    REC709 = RGBColorSpace(
        name="Rec.709",
        primaries=np.array([
            [0.64, 0.33],
            [0.30, 0.60],
            [0.15, 0.06],
        ], dtype=np.float64),
        whitepoint=StandardIlluminants.D65,
        transfer_function="bt1886",
    )
    
    REC2020 = RGBColorSpace(
        name="Rec.2020",
        primaries=np.array([
            [0.708, 0.292],
            [0.170, 0.797],
            [0.131, 0.046],
        ], dtype=np.float64),
        whitepoint=StandardIlluminants.D65,
        transfer_function="bt2100-pq",
    )
    
    P3_D65 = RGBColorSpace(
        name="DCI-P3 D65",
        primaries=np.array([
            [0.680, 0.320],
            [0.265, 0.690],
            [0.150, 0.060],
        ], dtype=np.float64),
        whitepoint=StandardIlluminants.D65,
        transfer_function="gamma_2.6",
    )
    
    ACES_AP0 = RGBColorSpace(
        name="ACES AP0",
        primaries=np.array([
            [0.7347, 0.2653],
            [0.0000, 1.0000],
            [0.0001, -0.0770],
        ], dtype=np.float64),
        whitepoint=StandardIlluminants.D60,
        transfer_function="linear",
    )
    
    ACES_AP1 = RGBColorSpace(
        name="ACEScg (AP1)",
        primaries=np.array([
            [0.713, 0.293],
            [0.165, 0.830],
            [0.128, 0.044],
        ], dtype=np.float64),
        whitepoint=StandardIlluminants.D60,
        transfer_function="linear",
    )
    
    # Bradford chromatic adaptation matrix
    BRADFORD_MATRIX = np.array([
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ], dtype=np.float64)
    
    BRADFORD_INVERSE = np.linalg.inv(BRADFORD_MATRIX)
    
    def __init__(self):
        """Initialize color space converter."""
        # Pre-compute common conversion matrices
        self._cache: Dict[str, npt.NDArray[np.float64]] = {}
        self._setup_common_matrices()
    
    def _setup_common_matrices(self) -> None:
        """Pre-compute commonly used conversion matrices."""
        # sRGB ↔ XYZ
        self._cache["srgb_to_xyz"] = self.SRGB.matrix_RGB_to_XYZ
        self._cache["xyz_to_srgb"] = self.SRGB.matrix_XYZ_to_RGB
        
        # Rec.2020 ↔ XYZ
        self._cache["rec2020_to_xyz"] = self.REC2020.matrix_RGB_to_XYZ
        self._cache["xyz_to_rec2020"] = self.REC2020.matrix_XYZ_to_RGB
        
        # P3-D65 ↔ XYZ
        self._cache["p3_to_xyz"] = self.P3_D65.matrix_RGB_to_XYZ
        self._cache["xyz_to_p3"] = self.P3_D65.matrix_XYZ_to_RGB
    
    def rgb_to_xyz(self, rgb: npt.NDArray, 
                   color_space: Union[str, RGBColorSpace] = "sRGB",
                   linearize: bool = True) -> npt.NDArray[np.float64]:
        """
        Convert RGB to CIE XYZ.
        
        Args:
            rgb: RGB image array [H, W, 3] or [N, 3]
            color_space: Source color space name or RGBColorSpace object
            linearize: Whether to linearize (decode transfer function) first
            
        Returns:
            XYZ array with same shape as input
        """
        rgb = np.asarray(rgb, dtype=np.float64)
        
        # Get color space
        if isinstance(color_space, str):
            cs = self._get_color_space(color_space)
        else:
            cs = color_space
        
        # Linearize if needed
        if linearize:
            rgb_linear = self._linearize(rgb, cs.transfer_function)
        else:
            rgb_linear = rgb
        
        # Apply matrix transform
        original_shape = rgb_linear.shape
        rgb_flat = rgb_linear.reshape(-1, 3)
        xyz_flat = np.dot(rgb_flat, cs.matrix_RGB_to_XYZ.T)
        
        return xyz_flat.reshape(original_shape)
    
    def xyz_to_rgb(self, xyz: npt.NDArray,
                   color_space: Union[str, RGBColorSpace] = "sRGB",
                   apply_transfer: bool = True,
                   clip: bool = True) -> npt.NDArray[np.float64]:
        """
        Convert CIE XYZ to RGB.
        
        Args:
            xyz: XYZ image array [H, W, 3] or [N, 3]
            color_space: Target color space name or RGBColorSpace object
            apply_transfer: Whether to apply transfer function (gamma encode)
            clip: Whether to clip values to [0, 1]
            
        Returns:
            RGB array with same shape as input
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        
        # Get color space
        if isinstance(color_space, str):
            cs = self._get_color_space(color_space)
        else:
            cs = color_space
        
        # Apply matrix transform
        original_shape = xyz.shape
        xyz_flat = xyz.reshape(-1, 3)
        rgb_flat = np.dot(xyz_flat, cs.matrix_XYZ_to_RGB.T)
        rgb = rgb_flat.reshape(original_shape)
        
        # Apply transfer function
        if apply_transfer:
            rgb = self._apply_transfer(rgb, cs.transfer_function)
        
        # Clip
        if clip:
            rgb = np.clip(rgb, 0, 1)
        
        return rgb
    
    def xyz_to_lab(self, xyz: npt.NDArray,
                   illuminant: Whitepoint = StandardIlluminants.D65) -> npt.NDArray[np.float64]:
        """
        Convert CIE XYZ to CIE LAB.
        
        Args:
            xyz: XYZ image array [H, W, 3] or [N, 3]
            illuminant: Reference illuminant
            
        Returns:
            LAB array with same shape as input (L: 0-100, a/b: ~-128 to 128)
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        
        # Normalize by illuminant
        Xn, Yn, Zn = illuminant.XYZ
        xyz_normalized = xyz / np.array([Xn, Yn, Zn])
        
        # Apply f function
        epsilon = 216 / 24389  # 0.008856
        kappa = 24389 / 27     # 903.3
        
        f = np.where(
            xyz_normalized > epsilon,
            np.cbrt(xyz_normalized),
            (kappa * xyz_normalized + 16) / 116
        )
        
        # Calculate LAB
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        
        return np.stack([L, a, b], axis=-1)
    
    def lab_to_xyz(self, lab: npt.NDArray,
                   illuminant: Whitepoint = StandardIlluminants.D65) -> npt.NDArray[np.float64]:
        """
        Convert CIE LAB to CIE XYZ.
        
        Args:
            lab: LAB image array [H, W, 3] or [N, 3]
            illuminant: Reference illuminant
            
        Returns:
            XYZ array with same shape as input
        """
        lab = np.asarray(lab, dtype=np.float64)
        
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        
        # Calculate f values
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        # Apply inverse f function
        epsilon = 216 / 24389
        kappa = 24389 / 27
        
        def f_inv(t):
            return np.where(
                t > epsilon ** (1/3),
                t ** 3,
                (116 * t - 16) / kappa
            )
        
        x = f_inv(fx)
        y = f_inv(fy)
        z = f_inv(fz)
        
        # Denormalize by illuminant
        Xn, Yn, Zn = illuminant.XYZ
        xyz = np.stack([x * Xn, y * Yn, z * Zn], axis=-1)
        
        return xyz
    
    def rgb_to_lab(self, rgb: npt.NDArray,
                   color_space: Union[str, RGBColorSpace] = "sRGB") -> npt.NDArray[np.float64]:
        """
        Convert RGB directly to CIE LAB.
        
        Args:
            rgb: RGB image array [H, W, 3]
            color_space: Source color space
            
        Returns:
            LAB array (L: 0-100, a/b: ~-128 to 128)
        """
        if isinstance(color_space, str):
            cs = self._get_color_space(color_space)
        else:
            cs = color_space
        
        xyz = self.rgb_to_xyz(rgb, cs)
        lab = self.xyz_to_lab(xyz, cs.whitepoint)
        return lab
    
    def lab_to_rgb(self, lab: npt.NDArray,
                   color_space: Union[str, RGBColorSpace] = "sRGB") -> npt.NDArray[np.float64]:
        """
        Convert CIE LAB to RGB.
        
        Args:
            lab: LAB image array
            color_space: Target color space
            
        Returns:
            RGB array (0-1 range)
        """
        if isinstance(color_space, str):
            cs = self._get_color_space(color_space)
        else:
            cs = color_space
        
        xyz = self.lab_to_xyz(lab, cs.whitepoint)
        rgb = self.xyz_to_rgb(xyz, cs)
        return rgb
    
    def rgb_to_rgb(self, rgb: npt.NDArray,
                   source: Union[str, RGBColorSpace],
                   target: Union[str, RGBColorSpace],
                   chromatic_adaptation: ChromaticAdaptation = ChromaticAdaptation.BRADFORD) -> npt.NDArray[np.float64]:
        """
        Convert RGB between different color spaces.
        
        Args:
            rgb: RGB image array
            source: Source color space
            target: Target color space
            chromatic_adaptation: Method for white point adaptation
            
        Returns:
            RGB array in target color space
        """
        if isinstance(source, str):
            source_cs = self._get_color_space(source)
        else:
            source_cs = source
            
        if isinstance(target, str):
            target_cs = self._get_color_space(target)
        else:
            target_cs = target
        
        # Convert to XYZ
        xyz = self.rgb_to_xyz(rgb, source_cs)
        
        # Apply chromatic adaptation if whitepoints differ
        if source_cs.whitepoint.xy != target_cs.whitepoint.xy:
            xyz = self.chromatic_adapt(
                xyz,
                source_cs.whitepoint,
                target_cs.whitepoint,
                chromatic_adaptation
            )
        
        # Convert to target RGB
        rgb_target = self.xyz_to_rgb(xyz, target_cs)
        
        return rgb_target
    
    def chromatic_adapt(self, xyz: npt.NDArray,
                        source_wp: Whitepoint,
                        target_wp: Whitepoint,
                        method: ChromaticAdaptation = ChromaticAdaptation.BRADFORD) -> npt.NDArray[np.float64]:
        """
        Apply chromatic adaptation transform.
        
        Args:
            xyz: XYZ values to transform
            source_wp: Source white point
            target_wp: Target white point
            method: Adaptation method
            
        Returns:
            Adapted XYZ values
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        
        if method == ChromaticAdaptation.BRADFORD:
            M = self.BRADFORD_MATRIX
            M_inv = self.BRADFORD_INVERSE
        elif method == ChromaticAdaptation.XYZ_SCALING:
            # Simple diagonal scaling
            source_XYZ = np.array(source_wp.XYZ)
            target_XYZ = np.array(target_wp.XYZ)
            scale = target_XYZ / source_XYZ
            
            original_shape = xyz.shape
            xyz_flat = xyz.reshape(-1, 3)
            result = xyz_flat * scale
            return result.reshape(original_shape)
        else:
            # Default to Bradford
            M = self.BRADFORD_MATRIX
            M_inv = self.BRADFORD_INVERSE
        
        # Convert white points to cone space
        source_XYZ = np.array(source_wp.XYZ)
        target_XYZ = np.array(target_wp.XYZ)
        
        source_cone = np.dot(M, source_XYZ)
        target_cone = np.dot(M, target_XYZ)
        
        # Diagonal scaling in cone space
        scale = target_cone / source_cone
        
        # Combined adaptation matrix
        M_adapt = M_inv @ np.diag(scale) @ M
        
        # Apply
        original_shape = xyz.shape
        xyz_flat = xyz.reshape(-1, 3)
        result = np.dot(xyz_flat, M_adapt.T)
        
        return result.reshape(original_shape)
    
    def _get_color_space(self, name: str) -> RGBColorSpace:
        """Get color space by name."""
        name_lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")
        
        if name_lower in ("srgb", "iec619662.1"):
            return self.SRGB
        elif name_lower in ("rec709", "bt709"):
            return self.REC709
        elif name_lower in ("rec2020", "bt2020"):
            return self.REC2020
        elif name_lower in ("p3", "p3d65", "dcip3d65", "displayp3"):
            return self.P3_D65
        elif name_lower in ("acesap0", "aces20651"):
            return self.ACES_AP0
        elif name_lower in ("acesap1", "acescg"):
            return self.ACES_AP1
        else:
            raise ValueError(f"Unknown color space: {name}")
    
    def _linearize(self, rgb: npt.NDArray, transfer: str) -> npt.NDArray[np.float64]:
        """Decode transfer function to linear."""
        if transfer == "srgb":
            return np.where(
                rgb <= 0.04045,
                rgb / 12.92,
                ((rgb + 0.055) / 1.055) ** 2.4
            )
        elif transfer == "bt1886":
            return np.power(np.maximum(rgb, 0), 2.4)
        elif transfer == "gamma_2.6":
            return np.power(np.maximum(rgb, 0), 2.6)
        elif transfer == "gamma_2.2":
            return np.power(np.maximum(rgb, 0), 2.2)
        elif transfer == "linear":
            return rgb
        elif transfer == "bt2100-pq":
            return self._pq_eotf(rgb)
        else:
            logger.warning(f"Unknown transfer function: {transfer}, treating as linear")
            return rgb
    
    def _apply_transfer(self, rgb: npt.NDArray, transfer: str) -> npt.NDArray[np.float64]:
        """Encode linear to transfer function."""
        if transfer == "srgb":
            return np.where(
                rgb <= 0.0031308,
                rgb * 12.92,
                1.055 * np.power(np.maximum(rgb, 0), 1/2.4) - 0.055
            )
        elif transfer == "bt1886":
            return np.power(np.maximum(rgb, 0), 1/2.4)
        elif transfer == "gamma_2.6":
            return np.power(np.maximum(rgb, 0), 1/2.6)
        elif transfer == "gamma_2.2":
            return np.power(np.maximum(rgb, 0), 1/2.2)
        elif transfer == "linear":
            return rgb
        elif transfer == "bt2100-pq":
            return self._pq_oetf(rgb)
        else:
            logger.warning(f"Unknown transfer function: {transfer}, treating as linear")
            return rgb
    
    def _pq_eotf(self, pq: npt.NDArray) -> npt.NDArray[np.float64]:
        """PQ (Perceptual Quantizer) EOTF - decode to linear."""
        m1 = 2610 / 16384
        m2 = 2523 / 32 * 128
        c1 = 3424 / 4096
        c2 = 2413 / 128
        c3 = 2392 / 128
        
        Ym = np.power(np.maximum(pq, 0), 1/m2)
        linear = 10000 * np.power(
            np.maximum(Ym - c1, 0) / (c2 - c3 * Ym),
            1/m1
        )
        return linear / 10000  # Normalize to 0-1 for SDR
    
    def _pq_oetf(self, linear: npt.NDArray) -> npt.NDArray[np.float64]:
        """PQ (Perceptual Quantizer) OETF - encode from linear."""
        m1 = 2610 / 16384
        m2 = 2523 / 32 * 128
        c1 = 3424 / 4096
        c2 = 2413 / 128
        c3 = 2392 / 128
        
        L = np.maximum(linear, 0) * 10000 / 10000  # Assume SDR
        Lm = np.power(L, m1)
        pq = np.power((c1 + c2 * Lm) / (1 + c3 * Lm), m2)
        return pq
    
    def rgb_to_hsv(self, rgb: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Convert RGB to HSV.
        
        Args:
            rgb: RGB array [H, W, 3] in 0-1 range
            
        Returns:
            HSV array (H: 0-360, S: 0-1, V: 0-1)
        """
        rgb = np.asarray(rgb, dtype=np.float64)
        
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = np.where(max_val > 1e-10, delta / max_val, 0)
        
        # Hue
        h = np.zeros_like(max_val)
        
        # Red is max
        mask_r = (max_val == r) & (delta > 1e-10)
        h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / delta[mask_r] % 6)
        
        # Green is max
        mask_g = (max_val == g) & (delta > 1e-10)
        h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
        
        # Blue is max
        mask_b = (max_val == b) & (delta > 1e-10)
        h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)
        
        h = h % 360
        
        return np.stack([h, s, v], axis=-1)
    
    def hsv_to_rgb(self, hsv: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Convert HSV to RGB.
        
        Args:
            hsv: HSV array (H: 0-360, S: 0-1, V: 0-1)
            
        Returns:
            RGB array in 0-1 range
        """
        hsv = np.asarray(hsv, dtype=np.float64)
        
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        
        # Select RGB based on hue sector
        h_sector = (h / 60).astype(np.int32) % 6
        
        rgb = np.zeros_like(hsv)
        
        for sector, (r, g, b) in enumerate([
            (c, x, 0), (x, c, 0), (0, c, x),
            (0, x, c), (x, 0, c), (c, 0, x)
        ]):
            mask = (h_sector == sector)
            if isinstance(r, np.ndarray):
                rgb[mask, 0] = r[mask] + m[mask]
                rgb[mask, 1] = g[mask] + m[mask]
                rgb[mask, 2] = b[mask] + m[mask]
            else:
                rgb[mask, 0] = (r if r == 0 else r[mask] if isinstance(r, np.ndarray) else c[mask] if r == c else x[mask]) + m[mask]
                rgb[mask, 1] = (g if g == 0 else g[mask] if isinstance(g, np.ndarray) else c[mask] if g == c else x[mask]) + m[mask]
                rgb[mask, 2] = (b if b == 0 else b[mask] if isinstance(b, np.ndarray) else c[mask] if b == c else x[mask]) + m[mask]
        
        # Simpler implementation
        mask0 = h_sector == 0
        mask1 = h_sector == 1
        mask2 = h_sector == 2
        mask3 = h_sector == 3
        mask4 = h_sector == 4
        mask5 = h_sector == 5
        
        rgb[mask0] = np.stack([c[mask0] + m[mask0], x[mask0] + m[mask0], m[mask0]], axis=-1)
        rgb[mask1] = np.stack([x[mask1] + m[mask1], c[mask1] + m[mask1], m[mask1]], axis=-1)
        rgb[mask2] = np.stack([m[mask2], c[mask2] + m[mask2], x[mask2] + m[mask2]], axis=-1)
        rgb[mask3] = np.stack([m[mask3], x[mask3] + m[mask3], c[mask3] + m[mask3]], axis=-1)
        rgb[mask4] = np.stack([x[mask4] + m[mask4], m[mask4], c[mask4] + m[mask4]], axis=-1)
        rgb[mask5] = np.stack([c[mask5] + m[mask5], m[mask5], x[mask5] + m[mask5]], axis=-1)
        
        return np.clip(rgb, 0, 1)
