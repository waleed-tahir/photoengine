"""
Parametric Optimizer - Advanced optimization strategies
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Callable, List
from scipy.optimize import differential_evolution, minimize
import logging

from .parameters import ParametricParameters, get_parameter_bounds

logger = logging.getLogger(__name__)


class ParametricOptimizer:
    """Advanced optimization strategies for parametric matching."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'population_size': 20,
            'max_generations': 100,
            'mutation': (0.5, 1.0),
            'recombination': 0.7
        }
    
    def optimize_global(self, loss_fn: Callable, x0: Optional[npt.NDArray] = None) -> Dict:
        """Global optimization using differential evolution."""
        lower, upper = get_parameter_bounds()
        bounds = list(zip(lower, upper))
        
        result = differential_evolution(
            loss_fn,
            bounds=bounds,
            x0=x0,
            maxiter=self.config['max_generations'],
            popsize=self.config['population_size'],
            mutation=self.config['mutation'],
            recombination=self.config['recombination'],
            updating='deferred',
            workers=-1,
            disp=True
        )
        
        return {
            'parameters': ParametricParameters.from_vector(result.x),
            'success': result.success,
            'iterations': result.nit,
            'loss': result.fun
        }
    
    def optimize_progressive(self, loss_fn: Callable, 
                            stages: List[str] = None) -> Dict:
        """
        Progressive optimization - optimize subsets of parameters sequentially.
        
        Stages: matrix -> curves -> tables
        """
        stages = stages or ['matrix', 'curves', 'tables']
        
        params = ParametricParameters.identity()
        
        for stage in stages:
            logger.info(f"Optimizing stage: {stage}")
            
            if stage == 'matrix':
                params = self._optimize_matrix(loss_fn, params)
            elif stage == 'curves':
                params = self._optimize_curves(loss_fn, params)
            elif stage == 'tables':
                params = self._optimize_tables(loss_fn, params)
        
        return {'parameters': params}
    
    def _optimize_matrix(self, loss_fn: Callable, 
                        params: ParametricParameters) -> ParametricParameters:
        """Optimize only the 3x4 matrix."""
        x0 = params.matrix.flatten()
        
        def matrix_loss(x):
            p = ParametricParameters.from_vector(params.to_vector())
            p.matrix = x.reshape(3, 4)
            return loss_fn(p.to_vector())
        
        bounds = [(-2, 2)] * 9 + [(-0.5, 0.5)] * 3
        result = minimize(matrix_loss, x0, method='L-BFGS-B', bounds=bounds)
        
        params.matrix = result.x.reshape(3, 4)
        return params
    
    def _optimize_curves(self, loss_fn: Callable,
                        params: ParametricParameters) -> ParametricParameters:
        """Optimize RGB and master curves."""
        x0 = np.concatenate([
            params.rgb_curves['R'],
            params.rgb_curves['G'],
            params.rgb_curves['B'],
            params.master_curve
        ])
        
        def curves_loss(x):
            p = ParametricParameters.from_vector(params.to_vector())
            p.rgb_curves['R'] = x[0:5]
            p.rgb_curves['G'] = x[5:10]
            p.rgb_curves['B'] = x[10:15]
            p.master_curve = x[15:20]
            return loss_fn(p.to_vector())
        
        bounds = [(0, 1)] * 20
        result = minimize(curves_loss, x0, method='L-BFGS-B', bounds=bounds)
        
        params.rgb_curves['R'] = result.x[0:5]
        params.rgb_curves['G'] = result.x[5:10]
        params.rgb_curves['B'] = result.x[10:15]
        params.master_curve = result.x[15:20]
        return params
    
    def _optimize_tables(self, loss_fn: Callable,
                        params: ParametricParameters) -> ParametricParameters:
        """Optimize 2D hue tables."""
        x0 = np.concatenate([params.hue_vs_sat.flatten(), params.hue_vs_hue.flatten()])
        
        def tables_loss(x):
            p = ParametricParameters.from_vector(params.to_vector())
            p.hue_vs_sat = x[:64].reshape(8, 8)
            p.hue_vs_hue = x[64:].reshape(8, 8)
            return loss_fn(p.to_vector())
        
        bounds = [(0.5, 2.0)] * 64 + [(-0.5, 0.5)] * 64
        result = minimize(tables_loss, x0, method='L-BFGS-B', bounds=bounds)
        
        params.hue_vs_sat = result.x[:64].reshape(8, 8)
        params.hue_vs_hue = result.x[64:].reshape(8, 8)
        return params
