"""Chromatica Pro Matching Engines

Provides parametric, neural, and hybrid color matching engines.
Neural and hybrid engines require PyTorch (optional dependency).
"""

# Core parametric engine (always available)
from src.engines.parametric.solver import ParametricSolver
from src.engines.parametric.optimizer import ParametricOptimizer
from src.engines.parametric.parameters import ParametricParameters, CDLParameters

# Optional imports for neural/hybrid engines (require PyTorch)
try:
    from src.engines.hybrid.orchestrator import HybridOrchestrator
    HAS_HYBRID = True
except ImportError:
    HybridOrchestrator = None
    HAS_HYBRID = False

try:
    from src.engines.neural.inference import NeuralInference
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
