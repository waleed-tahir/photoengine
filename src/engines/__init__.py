"""Chromatica Pro Matching Engines

Provides parametric, neural, and hybrid color matching engines.
Neural and hybrid engines require PyTorch (optional dependency).
"""

# Core parametric engine (always available)
from .parametric.solver import ParametricSolver
from .parametric.optimizer import ParametricOptimizer
from .parametric.parameters import ParametricParameters, CDLParameters

# Optional imports for neural/hybrid engines (require PyTorch)
try:
    from .hybrid.orchestrator import HybridOrchestrator
    HAS_HYBRID = True
except ImportError:
    HybridOrchestrator = None
    HAS_HYBRID = False

try:
    from .neural.inference import NeuralInference
    HAS_NEURAL = True
except ImportError:
    NeuralInference = None
    HAS_NEURAL = False

__all__ = [
    "ParametricSolver",
    "ParametricOptimizer", 
    "ParametricParameters",
    "CDLParameters",
    "HybridOrchestrator",
    "NeuralInference",
    "HAS_HYBRID",
    "HAS_NEURAL",
]
