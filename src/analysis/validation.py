"""
ColorChecker Validation Suite

Professional validation against X-Rite ColorChecker chart for accuracy testing.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColorCheckerPatch:
    """Single ColorChecker patch reference data."""
    name: str
    row: int
    col: int
    lab_d50: Tuple[float, float, float]  # Reference LAB (D50)
    srgb: Tuple[float, float, float]      # Approximate sRGB


# X-Rite ColorChecker Classic (24 patches) - CIE LAB D50 reference values
COLORCHECKER_CLASSIC = [
    ColorCheckerPatch("Dark Skin", 0, 0, (37.54, 14.37, 14.92), (115, 82, 68)),
    ColorCheckerPatch("Light Skin", 0, 1, (65.71, 18.13, 17.81), (194, 150, 130)),
    ColorCheckerPatch("Blue Sky", 0, 2, (49.93, -4.88, -21.93), (98, 122, 157)),
    ColorCheckerPatch("Foliage", 0, 3, (43.14, -13.10, 21.91), (87, 108, 67)),
    ColorCheckerPatch("Blue Flower", 0, 4, (55.11, 8.84, -25.40), (133, 128, 177)),
    ColorCheckerPatch("Bluish Green", 0, 5, (70.72, -33.40, -0.20), (103, 189, 170)),
    ColorCheckerPatch("Orange", 1, 0, (62.66, 36.07, 57.10), (214, 126, 44)),
    ColorCheckerPatch("Purplish Blue", 1, 1, (40.02, 10.41, -45.96), (80, 91, 166)),
    ColorCheckerPatch("Moderate Red", 1, 2, (51.12, 48.24, 16.25), (193, 90, 99)),
    ColorCheckerPatch("Purple", 1, 3, (30.33, 22.98, -21.59), (94, 60, 108)),
    ColorCheckerPatch("Yellow Green", 1, 4, (72.53, -23.71, 57.26), (157, 188, 64)),
    ColorCheckerPatch("Orange Yellow", 1, 5, (71.94, 19.36, 67.86), (224, 163, 46)),
    ColorCheckerPatch("Blue", 2, 0, (28.78, 14.18, -50.30), (56, 61, 150)),
    ColorCheckerPatch("Green", 2, 1, (55.26, -38.34, 31.37), (70, 148, 73)),
    ColorCheckerPatch("Red", 2, 2, (42.10, 53.38, 28.19), (175, 54, 60)),
    ColorCheckerPatch("Yellow", 2, 3, (81.73, 4.04, 79.82), (231, 199, 31)),
    ColorCheckerPatch("Magenta", 2, 4, (51.94, 49.99, -14.57), (187, 86, 149)),
    ColorCheckerPatch("Cyan", 2, 5, (51.04, -28.63, -28.64), (8, 133, 161)),
    ColorCheckerPatch("White", 3, 0, (96.54, -0.43, 1.19), (243, 243, 242)),
    ColorCheckerPatch("Neutral 8", 3, 1, (81.26, -0.64, -0.34), (200, 200, 200)),
    ColorCheckerPatch("Neutral 6.5", 3, 2, (66.77, -0.73, -0.50), (160, 160, 160)),
    ColorCheckerPatch("Neutral 5", 3, 3, (50.87, -0.15, -0.27), (122, 122, 121)),
    ColorCheckerPatch("Neutral 3.5", 3, 4, (35.66, -0.42, -1.23), (85, 85, 85)),
    ColorCheckerPatch("Black", 3, 5, (20.46, -0.08, -0.97), (52, 52, 52)),
]


@dataclass 
class ValidationResult:
    """Validation result against ColorChecker."""
    mean_delta_e: float
    max_delta_e: float
    min_delta_e: float
    std_delta_e: float
    patches_under_1: int
    patches_under_2: int
    patches_under_3: int
    patch_results: List[Dict]
    passed: bool


class ColorCheckerValidator:
    """
    Validate color matching against ColorChecker reference.
    
    Professional standard: ΔE00 < 1.0 for imperceptible difference.
    
    Example:
        >>> validator = ColorCheckerValidator()
        >>> result = validator.validate(matched_patches, reference_patches)
        >>> print(f"Mean ΔE00: {result.mean_delta_e:.2f}")
    """
    
    def __init__(self, threshold_pass: float = 2.0):
        self.threshold_pass = threshold_pass
        self.reference_patches = COLORCHECKER_CLASSIC
    
    def validate(self, matched_lab: npt.NDArray, 
                 reference_lab: Optional[npt.NDArray] = None) -> ValidationResult:
        """
        Validate matched image against reference.
        
        Args:
            matched_lab: Matched patch LAB values [24, 3]
            reference_lab: Optional reference LAB (uses ColorChecker if None)
            
        Returns:
            ValidationResult with detailed metrics
        """
        from ..core.color_science import ColorScience
        
        if reference_lab is None:
            reference_lab = np.array([p.lab_d50 for p in self.reference_patches])
        
        matched_lab = np.asarray(matched_lab)
        reference_lab = np.asarray(reference_lab)
        
        # Calculate ΔE2000 for each patch
        delta_e = ColorScience.delta_e_2000(matched_lab, reference_lab)
        
        # Per-patch results
        patch_results = []
        for i, patch in enumerate(self.reference_patches):
            patch_results.append({
                'name': patch.name,
                'delta_e': float(delta_e[i]),
                'matched_lab': matched_lab[i].tolist(),
                'reference_lab': reference_lab[i].tolist(),
                'passed': delta_e[i] < self.threshold_pass
            })
        
        return ValidationResult(
            mean_delta_e=float(np.mean(delta_e)),
            max_delta_e=float(np.max(delta_e)),
            min_delta_e=float(np.min(delta_e)),
            std_delta_e=float(np.std(delta_e)),
            patches_under_1=int(np.sum(delta_e < 1.0)),
            patches_under_2=int(np.sum(delta_e < 2.0)),
            patches_under_3=int(np.sum(delta_e < 3.0)),
            patch_results=patch_results,
            passed=float(np.mean(delta_e)) < self.threshold_pass
        )
    
    def extract_patches(self, image: npt.NDArray, 
                        patch_coords: Optional[List[Tuple[int, int, int, int]]] = None,
                        auto_detect: bool = True) -> npt.NDArray:
        """
        Extract ColorChecker patches from image.
        
        Args:
            image: Input image [H, W, 3]
            patch_coords: Optional list of (x, y, w, h) for each patch
            auto_detect: Attempt automatic detection
            
        Returns:
            Extracted patch colors [24, 3]
        """
        if patch_coords is not None:
            return self._extract_from_coords(image, patch_coords)
        
        if auto_detect:
            coords = self._detect_colorchecker(image)
            if coords is not None:
                return self._extract_from_coords(image, coords)
        
        # Fallback: assume standard grid layout
        return self._extract_grid(image, rows=4, cols=6)
    
    def _extract_from_coords(self, image: npt.NDArray,
                             coords: List[Tuple[int, int, int, int]]) -> npt.NDArray:
        """Extract patches from specified coordinates."""
        patches = []
        for x, y, w, h in coords:
            # Extract central region (avoid edges)
            margin = min(w, h) // 4
            patch = image[y+margin:y+h-margin, x+margin:x+w-margin]
            patches.append(np.mean(patch, axis=(0, 1)))
        return np.array(patches)
    
    def _extract_grid(self, image: npt.NDArray, 
                      rows: int = 4, cols: int = 6) -> npt.NDArray:
        """Extract patches assuming uniform grid."""
        h, w = image.shape[:2]
        patch_h = h // rows
        patch_w = w // cols
        
        patches = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * patch_h, (row + 1) * patch_h
                x1, x2 = col * patch_w, (col + 1) * patch_w
                
                # Central region
                margin_h = patch_h // 4
                margin_w = patch_w // 4
                patch = image[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w]
                patches.append(np.mean(patch, axis=(0, 1)))
        
        return np.array(patches)
    
    def _detect_colorchecker(self, image: npt.NDArray) -> Optional[List]:
        """Attempt automatic ColorChecker detection."""
        try:
            import cv2
            
            # Convert to uint8 for OpenCV
            img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            
            # Try OpenCV's ColorChecker detector if available
            if hasattr(cv2, 'mcc'):
                detector = cv2.mcc.CCheckerDetector_create()
                if detector.process(img_8bit, cv2.mcc.MCC24):
                    checker = detector.getBestColorChecker()
                    if checker is not None:
                        boxes = checker.getBox()
                        # Convert to (x, y, w, h) format
                        coords = []
                        for box in boxes:
                            x, y = int(box[0]), int(box[1])
                            w, h = int(box[2] - box[0]), int(box[3] - box[1])
                            coords.append((x, y, w, h))
                        return coords
            
            logger.warning("Automatic ColorChecker detection not available")
            return None
        except Exception as e:
            logger.warning(f"ColorChecker detection failed: {e}")
            return None
    
    def generate_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 60,
            "COLORCHECKER VALIDATION REPORT",
            "=" * 60,
            "",
            f"Overall Result: {'PASS ✓' if result.passed else 'FAIL ✗'}",
            "",
            "Summary Statistics:",
            f"  Mean ΔE00:  {result.mean_delta_e:.3f}",
            f"  Max ΔE00:   {result.max_delta_e:.3f}",
            f"  Min ΔE00:   {result.min_delta_e:.3f}",
            f"  Std ΔE00:   {result.std_delta_e:.3f}",
            "",
            "Patch Distribution:",
            f"  ΔE00 < 1.0 (imperceptible): {result.patches_under_1}/24",
            f"  ΔE00 < 2.0 (acceptable):    {result.patches_under_2}/24",
            f"  ΔE00 < 3.0 (noticeable):    {result.patches_under_3}/24",
            "",
            "Per-Patch Results:",
            "-" * 40,
        ]
        
        for patch in result.patch_results:
            status = "✓" if patch['passed'] else "✗"
            lines.append(f"  {status} {patch['name']}: ΔE00 = {patch['delta_e']:.2f}")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
