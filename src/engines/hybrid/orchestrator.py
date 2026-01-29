"""
Hybrid Matching Orchestrator

Intelligently combines parametric and neural engines for optimal results.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Literal, Callable
import logging
import time

from src.engines.parametric.solver import ParametricSolver
from src.engines.parametric.parameters import ParametricParameters

logger = logging.getLogger(__name__)


class HybridOrchestrator:
    """
    Intelligently orchestrates parametric and neural engines.
    
    Strategies:
    - 'parametric': Parametric only (best for simple corrections)
    - 'neural': Neural only (best for complex looks)
    - 'hybrid': Neural initialization + parametric refinement
    - 'auto': Automatically choose based on image analysis
    
    Example:
        >>> orchestrator = HybridOrchestrator()
        >>> result = orchestrator.match(source, target, strategy='hybrid')
    """
    
    def __init__(self, neural_model_path: Optional[str] = None):
        self.parametric = ParametricSolver()
        self.neural = None
        self.neural_model_path = neural_model_path
        
        logger.info("Hybrid orchestrator initialized")
    
    def _ensure_neural(self):
        """Lazy load neural engine."""
        if self.neural is None:
            try:
                from src.engines.neural.inference import NeuralInference
                self.neural = NeuralInference(self.neural_model_path)
            except Exception as e:
                logger.warning(f"Neural engine unavailable: {e}")
                return False
        return True
    
    def match(self, source: npt.NDArray, target: npt.NDArray,
              strategy: Literal['parametric', 'neural', 'hybrid', 'auto'] = 'auto',
              mask: Optional[npt.NDArray] = None,
              progress_cb: Optional[Callable[[int], None]] = None) -> Dict:
        """
        Match source to target using specified strategy.
        
        Args:
            source: Source image [H, W, 3]
            target: Target image [H, W, 3]
            strategy: Matching strategy
            mask: Optional mask for weighted matching
            
        Returns:
            Dictionary with 'output', 'parameters', 'metrics', 'strategy_used'
        """
        start_time = time.time()
        
        # Auto-select strategy
        if strategy == 'auto':
            strategy = self._auto_select_strategy(source, target)
        
        if progress_cb: progress_cb(20)
        logger.info(f"Using strategy: {strategy}")
        
        if strategy == 'parametric':
            result = self._match_parametric(source, target, mask, progress_cb=progress_cb)
        elif strategy == 'neural' and self._ensure_neural():
            result = self._match_neural(source, target)
        elif strategy == 'hybrid' and self._ensure_neural():
            if progress_cb: progress_cb(40)
            result = self._match_hybrid(source, target, mask)
        else:
            # Fallback to parametric
            result = self._match_parametric(source, target, mask)
        
        result['strategy_used'] = strategy
        result['total_time'] = time.time() - start_time
        
        return result
    
    def _auto_select_strategy(self, source: npt.NDArray, 
                              target: npt.NDArray) -> str:
        """Automatically select best strategy based on image analysis."""
        # Analyze color difference complexity
        source_mean = np.mean(source, axis=(0, 1))
        target_mean = np.mean(target, axis=(0, 1))
        mean_diff = np.linalg.norm(source_mean - target_mean)
        
        # Analyze histogram distributions
        source_std = np.std(source)
        target_std = np.std(target)
        std_diff = abs(source_std - target_std)
        
        # Simple corrections: parametric
        # Complex looks: hybrid
        complexity = mean_diff * 2 + std_diff * 5
        
        if complexity < 0.2:
            return 'parametric'
        elif complexity > 0.5:
            return 'hybrid'
        else:
            return 'parametric'
    
    def _match_parametric(self, source: npt.NDArray, target: npt.NDArray,
                          mask: Optional[npt.NDArray],
                          progress_cb: Optional[Callable[[int], None]] = None) -> Dict:
        """Pure parametric matching."""
        result = self.parametric.match(source, target, mask, progress_cb=progress_cb)
        
        return {
            'output': self.parametric.apply(source, result['parameters']),
            'parameters': result['parameters'],
            'metrics': result['metrics']
        }
    
    def _match_neural(self, source: npt.NDArray, target: npt.NDArray) -> Dict:
        """Pure neural matching."""
        params = self.neural.predict(source, target)
        output = self.parametric.apply(source, params)
        
        from src.core.color_science import ColorScience
        metrics = ColorScience.calculate_metrics(output, target)
        
        return {
            'output': output,
            'parameters': params,
            'metrics': {'delta_e_mean': metrics.delta_e_mean}
        }
    
    def _match_hybrid(self, source: npt.NDArray, target: npt.NDArray,
                      mask: Optional[npt.NDArray]) -> Dict:
        """Neural initialization + parametric refinement."""
        # Step 1: Neural prediction for initial parameters
        initial_params = self.neural.predict(source, target)
        
        # Step 2: Refine with parametric optimization
        result = self.parametric.match(
            source, target, mask, initial_params=initial_params,
            progress_cb=progress_cb
        )
        
        return {
            'output': self.parametric.apply(source, result['parameters']),
            'parameters': result['parameters'],
            'metrics': result['metrics'],
            'initial_params': initial_params
        }
    
    def quick_match(self, source: npt.NDArray, target: npt.NDArray) -> Dict:
        """Fast CDL-only matching for real-time preview."""
        result = self.parametric.optimize_cdl_only(source, target)
        return {
            'output': self.parametric._apply_cdl(source, result['cdl']),
            'parameters': result['parameters'],
            'metrics': result['metrics']
        }
