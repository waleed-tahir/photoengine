"""
Quality Analysis and Metrics

Professional image quality assessment for color matching.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive quality analysis report."""
    # Color accuracy
    delta_e_mean: float
    delta_e_max: float
    delta_e_p95: float
    
    # Signal quality
    psnr: float
    mse: float
    ssim: Optional[float]
    
    # Technical
    clipping_low_pct: float
    clipping_high_pct: float
    dynamic_range: float
    
    # Grading
    overall_grade: str  # A, B, C, D, F
    passed: bool


class QualityAnalyzer:
    """
    Comprehensive image quality analyzer.
    
    Provides professional metrics for validating color matching results.
    """
    
    # Thresholds for grading
    GRADE_THRESHOLDS = {
        'A': {'delta_e': 1.0, 'psnr': 45},
        'B': {'delta_e': 2.0, 'psnr': 40},
        'C': {'delta_e': 3.0, 'psnr': 35},
        'D': {'delta_e': 5.0, 'psnr': 30},
    }
    
    def analyze(self, result: npt.NDArray, target: npt.NDArray) -> QualityReport:
        """
        Analyze match quality.
        
        Args:
            result: Matched image [H, W, 3]
            target: Target image [H, W, 3]
            
        Returns:
            QualityReport with all metrics
        """
        from ..core.color_science import ColorScience
        
        result = np.asarray(result, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        
        # Color metrics
        metrics = ColorScience.calculate_metrics(result, target)
        
        # Clipping analysis
        clipping = ColorScience.calculate_gamut_clipping(result)
        
        # Dynamic range
        dr = self._calculate_dynamic_range(result)
        
        # SSIM (if available)
        ssim = self._calculate_ssim(result, target)
        
        # Calculate grade
        grade = self._calculate_grade(metrics.delta_e_mean, metrics.psnr)
        
        return QualityReport(
            delta_e_mean=metrics.delta_e_mean,
            delta_e_max=metrics.delta_e_max,
            delta_e_p95=metrics.delta_e_percentile_95,
            psnr=metrics.psnr,
            mse=metrics.mse,
            ssim=ssim,
            clipping_low_pct=clipping['clipped_low_percent'],
            clipping_high_pct=clipping['clipped_high_percent'],
            dynamic_range=dr,
            overall_grade=grade,
            passed=grade in ['A', 'B', 'C']
        )
    
    def _calculate_dynamic_range(self, image: npt.NDArray) -> float:
        """Calculate dynamic range in stops."""
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        
        if min_val <= 0 or max_val <= 0:
            return 0.0
        
        return np.log2(max_val / max(min_val, 1e-6))
    
    def _calculate_ssim(self, result: npt.NDArray, 
                        target: npt.NDArray) -> Optional[float]:
        """Calculate SSIM if available."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to same range
            result_8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            target_8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)
            
            return float(ssim(result_8, target_8, channel_axis=2))
        except ImportError:
            return None
    
    def _calculate_grade(self, delta_e: float, psnr: float) -> str:
        """Calculate overall grade based on metrics."""
        for grade, thresholds in self.GRADE_THRESHOLDS.items():
            if delta_e <= thresholds['delta_e'] and psnr >= thresholds['psnr']:
                return grade
        return 'F'
    
    def generate_report(self, report: QualityReport) -> str:
        """Generate human-readable quality report."""
        lines = [
            "=" * 50,
            "QUALITY ANALYSIS REPORT",
            "=" * 50,
            "",
            f"Overall Grade: {report.overall_grade} {'✓' if report.passed else '✗'}",
            "",
            "Color Accuracy:",
            f"  Mean ΔE00:  {report.delta_e_mean:.3f}",
            f"  Max ΔE00:   {report.delta_e_max:.3f}",
            f"  95th ΔE00:  {report.delta_e_p95:.3f}",
            "",
            "Signal Quality:",
            f"  PSNR:       {report.psnr:.2f} dB",
            f"  MSE:        {report.mse:.6f}",
        ]
        
        if report.ssim is not None:
            lines.append(f"  SSIM:       {report.ssim:.4f}")
        
        lines.extend([
            "",
            "Technical Analysis:",
            f"  Clipping (low):   {report.clipping_low_pct:.2f}%",
            f"  Clipping (high):  {report.clipping_high_pct:.2f}%",
            f"  Dynamic Range:    {report.dynamic_range:.1f} stops",
            "",
            "=" * 50
        ])
        
        return "\n".join(lines)
    
    def compare_pipelines(self, source: npt.NDArray, target: npt.NDArray,
                         results: Dict[str, npt.NDArray]) -> Dict:
        """
        Compare multiple matching pipelines.
        
        Args:
            source: Source image
            target: Target image
            results: Dict of {pipeline_name: result_image}
            
        Returns:
            Comparison results
        """
        comparisons = {}
        
        for name, result in results.items():
            report = self.analyze(result, target)
            comparisons[name] = {
                'grade': report.overall_grade,
                'delta_e_mean': report.delta_e_mean,
                'psnr': report.psnr,
                'passed': report.passed
            }
        
        # Rank by delta_e
        rankings = sorted(comparisons.items(), 
                         key=lambda x: x[1]['delta_e_mean'])
        
        return {
            'comparisons': comparisons,
            'rankings': [(name, data['delta_e_mean']) for name, data in rankings],
            'best': rankings[0][0] if rankings else None
        }
