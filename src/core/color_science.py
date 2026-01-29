"""
Color Science Utilities for Chromatica Pro

ΔE2000 calculation, histogram analysis, color chart detection, and quality metrics.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColorMetrics:
    """Color difference metrics between two images"""
    delta_e_mean: float
    delta_e_max: float
    delta_e_min: float
    delta_e_std: float
    delta_e_percentile_95: float
    psnr: float
    mse: float


class ColorScience:
    """
    Professional color science utilities.
    
    Provides accurate ΔE2000 calculation, histogram analysis,
    and quality metrics for color matching validation.
    """
    
    @staticmethod
    def delta_e_2000(lab1: npt.NDArray, lab2: npt.NDArray,
                     kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> npt.NDArray[np.float64]:
        """
        Calculate CIE ΔE2000 color difference.
        
        This is the industry standard for perceptual color difference.
        ΔE00 < 1.0 is imperceptible to most observers.
        
        Args:
            lab1: First LAB array [H, W, 3] or [N, 3]
            lab2: Second LAB array [H, W, 3] or [N, 3]
            kL, kC, kH: Parametric weighting factors
            
        Returns:
            ΔE2000 values with same shape as input (minus last dimension)
        """
        lab1 = np.asarray(lab1, dtype=np.float64)
        lab2 = np.asarray(lab2, dtype=np.float64)
        
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
        
        # Calculate C' and h'
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_avg = (C1 + C2) / 2
        
        G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
        
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        
        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
        
        # Differences
        dL_prime = L2 - L1
        dC_prime = C2_prime - C1_prime
        
        dh_prime = np.where(
            C1_prime * C2_prime == 0, 0,
            np.where(np.abs(h2_prime - h1_prime) <= 180, h2_prime - h1_prime,
                    np.where(h2_prime - h1_prime > 180, h2_prime - h1_prime - 360,
                            h2_prime - h1_prime + 360)))
        
        dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))
        
        # Weighted factors
        L_prime_avg = (L1 + L2) / 2
        C_prime_avg = (C1_prime + C2_prime) / 2
        
        h_prime_avg = np.where(
            C1_prime * C2_prime == 0, h1_prime + h2_prime,
            np.where(np.abs(h1_prime - h2_prime) <= 180, (h1_prime + h2_prime) / 2,
                    np.where(h1_prime + h2_prime < 360, (h1_prime + h2_prime + 360) / 2,
                            (h1_prime + h2_prime - 360) / 2)))
        
        T = (1 - 0.17 * np.cos(np.radians(h_prime_avg - 30)) +
             0.24 * np.cos(np.radians(2 * h_prime_avg)) +
             0.32 * np.cos(np.radians(3 * h_prime_avg + 6)) -
             0.20 * np.cos(np.radians(4 * h_prime_avg - 63)))
        
        dTheta = 30 * np.exp(-((h_prime_avg - 275) / 25)**2)
        R_C = 2 * np.sqrt(C_prime_avg**7 / (C_prime_avg**7 + 25**7))
        S_L = 1 + (0.015 * (L_prime_avg - 50)**2) / np.sqrt(20 + (L_prime_avg - 50)**2)
        S_C = 1 + 0.045 * C_prime_avg
        S_H = 1 + 0.015 * C_prime_avg * T
        R_T = -np.sin(np.radians(2 * dTheta)) * R_C
        
        # Final ΔE2000
        dE = np.sqrt(
            (dL_prime / (kL * S_L))**2 +
            (dC_prime / (kC * S_C))**2 +
            (dH_prime / (kH * S_H))**2 +
            R_T * (dC_prime / (kC * S_C)) * (dH_prime / (kH * S_H))
        )
        
        return dE
    
    @staticmethod
    def calculate_metrics(result: npt.NDArray, target: npt.NDArray,
                         converter=None) -> ColorMetrics:
        """
        Calculate comprehensive color difference metrics.
        
        Args:
            result: Result image [H, W, 3] in 0-1 RGB
            target: Target image [H, W, 3] in 0-1 RGB
            converter: Optional ColorSpaceConverter for LAB conversion
            
        Returns:
            ColorMetrics dataclass with all metrics
        """
        # Convert to LAB if converter provided
        if converter is not None:
            result_lab = converter.rgb_to_lab(result)
            target_lab = converter.rgb_to_lab(target)
        else:
            # Simple conversion via XYZ
            result_lab = ColorScience._simple_rgb_to_lab(result)
            target_lab = ColorScience._simple_rgb_to_lab(target)
        
        # Calculate ΔE2000
        delta_e = ColorScience.delta_e_2000(result_lab, target_lab)
        
        # MSE and PSNR
        mse = np.mean((result - target) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        return ColorMetrics(
            delta_e_mean=float(np.mean(delta_e)),
            delta_e_max=float(np.max(delta_e)),
            delta_e_min=float(np.min(delta_e)),
            delta_e_std=float(np.std(delta_e)),
            delta_e_percentile_95=float(np.percentile(delta_e, 95)),
            psnr=float(psnr),
            mse=float(mse)
        )
    
    @staticmethod
    def _simple_rgb_to_lab(rgb: npt.NDArray) -> npt.NDArray:
        """Simple sRGB to LAB conversion for ΔE calculation."""
        # Linearize sRGB
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        
        # sRGB to XYZ matrix
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
        xyz = np.dot(linear, M.T)
        
        # XYZ to LAB (D65 illuminant)
        xyz_n = np.array([0.95047, 1.0, 1.08883])
        xyz_normalized = xyz / xyz_n
        
        epsilon = 216 / 24389
        kappa = 24389 / 27
        f = np.where(xyz_normalized > epsilon, np.cbrt(xyz_normalized),
                    (kappa * xyz_normalized + 16) / 116)
        
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        
        return np.stack([L, a, b], axis=-1)
    
    @staticmethod
    def analyze_histogram(image: npt.NDArray, bins: int = 256) -> Dict:
        """
        Analyze image histogram for color space detection.
        
        Args:
            image: RGB image [H, W, 3]
            bins: Number of histogram bins
            
        Returns:
            Dictionary with histogram analysis results
        """
        flat = image.flatten()
        hist, bin_edges = np.histogram(flat, bins=bins, range=(0, 1))
        
        # Find peak
        peak_idx = np.argmax(hist)
        peak_position = peak_idx / bins
        
        # Calculate spread
        non_zero = hist > 0
        if np.any(non_zero):
            first_nz = np.argmax(non_zero)
            last_nz = bins - 1 - np.argmax(non_zero[::-1])
            spread = (last_nz - first_nz) / bins
        else:
            spread = 0.0
        
        # Per-channel stats
        channel_stats = []
        for c in range(min(3, image.shape[-1])):
            ch = image[..., c].flatten()
            channel_stats.append({
                'mean': float(np.mean(ch)),
                'std': float(np.std(ch)),
                'min': float(np.min(ch)),
                'max': float(np.max(ch))
            })
        
        return {
            'peak_position': float(peak_position),
            'spread': float(spread),
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'histogram': hist,
            'channels': channel_stats
        }
    
    @staticmethod
    def detect_colorchecker(image: npt.NDArray) -> Optional[Dict]:
        """
        Detect ColorChecker chart in image.
        
        Returns patch locations and extracted colors if found.
        """
        # Simplified detection - full implementation would use ML
        # This returns None indicating manual selection needed
        logger.warning("Automatic ColorChecker detection not implemented")
        return None
    
    @staticmethod
    def extract_skin_tones(image: npt.NDArray, 
                          threshold: float = 0.3) -> npt.NDArray:
        """
        Extract pixels likely to be skin tones.
        
        Uses simple heuristics based on RGB ratios common in skin.
        
        Args:
            image: RGB image [H, W, 3]
            threshold: Confidence threshold
            
        Returns:
            Boolean mask of skin pixels
        """
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        
        # Skin tone heuristics (works for diverse skin tones)
        # R > G > B pattern with specific ratios
        condition1 = (r > g) & (g > b)
        condition2 = (r - g) > 0.05
        condition3 = (r > 0.2) & (r < 0.95)
        condition4 = (g > 0.1) & (g < 0.85)
        condition5 = np.abs(r - g) < 0.4
        
        skin_mask = condition1 & condition2 & condition3 & condition4 & condition5
        
        return skin_mask
    
    @staticmethod
    def calculate_gamut_clipping(image: npt.NDArray) -> Dict:
        """
        Analyze gamut clipping in image.
        
        Returns:
            Dictionary with clipping statistics
        """
        total_pixels = image.size
        
        clipped_low = np.sum(image < 0)
        clipped_high = np.sum(image > 1)
        
        return {
            'clipped_low_percent': 100 * clipped_low / total_pixels,
            'clipped_high_percent': 100 * clipped_high / total_pixels,
            'total_clipped_percent': 100 * (clipped_low + clipped_high) / total_pixels,
            'in_gamut_percent': 100 * (1 - (clipped_low + clipped_high) / total_pixels)
        }
