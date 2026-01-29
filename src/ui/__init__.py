"""Chromatica Pro User Interface"""
from src.ui.main_window import ChromaticaMainWindow, main
from src.ui.viewport import GPUViewport
from src.ui.workflow import WorkflowManager, Project

__all__ = [
    "ChromaticaMainWindow", "main",
    "GPUViewport",
    "WorkflowManager", "Project"
]

# Optional imports for controls and scopes when available
try:
    from src.ui.controls import CDLControls, ControlPanel, CurveEditor
    __all__.extend(["CDLControls", "ControlPanel", "CurveEditor"])
except ImportError:
    pass

try:
    from src.ui.scopes import ScopesPanel, WaveformWidget, VectorscopeWidget
    __all__.extend(["ScopesPanel", "WaveformWidget", "VectorscopeWidget"])
except ImportError:
    pass
