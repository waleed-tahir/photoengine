"""Chromatica Pro User Interface"""
from .main_window import ChromaticaMainWindow, main
from .viewport import GPUViewport
from .workflow import WorkflowManager, Project

__all__ = [
    "ChromaticaMainWindow", "main",
    "GPUViewport",
    "WorkflowManager", "Project"
]

# Optional imports for controls and scopes when available
try:
    from .controls import CDLControls, ControlPanel, CurveEditor
    __all__.extend(["CDLControls", "ControlPanel", "CurveEditor"])
except ImportError:
    pass

try:
    from .scopes import ScopesPanel, WaveformWidget, VectorscopeWidget
    __all__.extend(["ScopesPanel", "WaveformWidget", "VectorscopeWidget"])
except ImportError:
    pass
