"""
Chromatica Pro Main Window

Professional Qt6 application with triple viewport, scopes, and controls.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Qt imports with fallback
try:
    from PySide6.QtWidgets import (
        QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QMenuBar, QToolBar, QStatusBar, QDockWidget,
        QFileDialog, QMessageBox, QProgressDialog, QLabel, QComboBox,
        QPushButton, QSlider
    )
    from PySide6.QtCore import Qt, Signal, Slot, QThread
    from PySide6.QtGui import QAction, QKeySequence, QImage, QPixmap
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logger.warning("PySide6 not installed. UI will not be available.")


if HAS_QT:
    class ImageViewport(QLabel):
        """Simple image viewport widget."""
        
        def __init__(self, title: str = ""):
            super().__init__()
            self.title = title
            self.image_data = None
            self.setMinimumSize(300, 200)
            self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
            self.setAlignment(Qt.AlignCenter)
            self.setText(title)
        
        def set_image(self, image: np.ndarray):
            """Display numpy image."""
            self.image_data = image
            
            # Convert to QImage
            h, w = image.shape[:2]
            image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            
            if image.ndim == 3:
                qimg = QImage(image_8bit.data, w, h, w * 3, QImage.Format_RGB888)
            else:
                qimg = QImage(image_8bit.data, w, h, w, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
    
    
    class MatchWorker(QThread):
        """Background worker for color matching."""
        progress = Signal(int)
        finished = Signal(dict)
        error = Signal(str)
        
        def __init__(self, source, target, mode='parametric'):
            super().__init__()
            self.source = source
            self.target = target
            self.mode = mode
        
        def run(self):
            try:
                from src.engines.hybrid.orchestrator import HybridOrchestrator
                
                orchestrator = HybridOrchestrator()
                self.progress.emit(10)
                
                result = orchestrator.match(
                    self.source, self.target, 
                    strategy=self.mode,
                    progress_cb=self.progress.emit
                )
                self.progress.emit(100)
                
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))
    
    
    class ChromaticaMainWindow(QMainWindow):
        """Professional main window for Chromatica Pro."""
        
        image_loaded = Signal(str)
        match_completed = Signal(dict)
        
        def __init__(self):
            super().__init__()
            
            self.setWindowTitle("Chromatica Pro v1.0")
            self.setGeometry(100, 100, 1600, 900)
            self.setStyleSheet(self._get_stylesheet())
            
            # State
            self.source_image = None
            self.reference_image = None
            self.result_image = None
            self.match_result = None
            
            self._setup_ui()
            self._setup_menu()
            self._setup_toolbar()
            self._setup_status_bar()
        
        def _get_stylesheet(self) -> str:
            return """
                QMainWindow { background-color: #1e1e1e; }
                QMenuBar { background-color: #2d2d2d; color: #e0e0e0; }
                QMenuBar::item:selected { background-color: #3d3d3d; }
                QMenu { background-color: #2d2d2d; color: #e0e0e0; }
                QMenu::item:selected { background-color: #0078d4; }
                QToolBar { background-color: #2d2d2d; border: none; padding: 4px; }
                QPushButton { 
                    background-color: #0078d4; color: white; 
                    border: none; padding: 8px 16px; border-radius: 4px;
                }
                QPushButton:hover { background-color: #1084d8; }
                QLabel { color: #e0e0e0; }
                QStatusBar { background-color: #2d2d2d; color: #e0e0e0; }
                QDockWidget { color: #e0e0e0; }
                QDockWidget::title { background-color: #2d2d2d; padding: 4px; }
                QComboBox { 
                    background-color: #3d3d3d; color: #e0e0e0;
                    border: 1px solid #555; padding: 4px;
                }
            """
        
        def _setup_ui(self):
            """Setup main UI."""
            central = QWidget()
            self.setCentralWidget(central)
            
            layout = QHBoxLayout(central)
            layout.setContentsMargins(8, 8, 8, 8)
            
            # Triple viewport splitter
            splitter = QSplitter(Qt.Horizontal)
            
            self.source_viewport = ImageViewport("Source")
            self.result_viewport = ImageViewport("Result")
            self.reference_viewport = ImageViewport("Reference")
            
            splitter.addWidget(self.source_viewport)
            splitter.addWidget(self.result_viewport)
            splitter.addWidget(self.reference_viewport)
            
            layout.addWidget(splitter)
        
        def _setup_menu(self):
            """Setup menu bar."""
            menubar = self.menuBar()
            
            # File menu
            file_menu = menubar.addMenu("&File")
            
            load_source = QAction("Load &Source...", self)
            load_source.setShortcut(QKeySequence.Open)
            load_source.triggered.connect(self.load_source)
            file_menu.addAction(load_source)
            
            load_ref = QAction("Load &Reference...", self)
            load_ref.setShortcut("Ctrl+R")
            load_ref.triggered.connect(self.load_reference)
            file_menu.addAction(load_ref)
            
            file_menu.addSeparator()
            
            export_menu = file_menu.addMenu("&Export")
            
            export_lut = QAction("Export 3D LUT (.cube)...", self)
            export_lut.triggered.connect(lambda: self.export("lut"))
            export_menu.addAction(export_lut)
            
            export_cdl = QAction("Export CDL...", self)
            export_cdl.triggered.connect(lambda: self.export("cdl"))
            export_menu.addAction(export_cdl)
            
            file_menu.addSeparator()
            
            exit_action = QAction("E&xit", self)
            exit_action.setShortcut(QKeySequence.Quit)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            # Edit menu
            edit_menu = menubar.addMenu("&Edit")
            
            match_action = QAction("&Match Colors", self)
            match_action.setShortcut("Ctrl+M")
            match_action.triggered.connect(self.match_colors)
            edit_menu.addAction(match_action)
            
            # Help menu
            help_menu = menubar.addMenu("&Help")
            
            about_action = QAction("&About", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
        
        def _setup_toolbar(self):
            """Setup toolbar."""
            toolbar = QToolBar("Main")
            toolbar.setMovable(False)
            self.addToolBar(toolbar)
            
            load_src_btn = QPushButton("Load Source")
            load_src_btn.clicked.connect(self.load_source)
            toolbar.addWidget(load_src_btn)
            
            load_ref_btn = QPushButton("Load Reference")
            load_ref_btn.clicked.connect(self.load_reference)
            toolbar.addWidget(load_ref_btn)
            
            toolbar.addSeparator()
            
            toolbar.addWidget(QLabel("Mode: "))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Parametric", "Neural", "Hybrid", "Auto"])
            toolbar.addWidget(self.mode_combo)
            
            match_btn = QPushButton("Match Colors")
            match_btn.clicked.connect(self.match_colors)
            toolbar.addWidget(match_btn)
        
        def _setup_status_bar(self):
            """Setup status bar."""
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage("Ready")
        
        @Slot()
        def load_source(self):
            """Load source image."""
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Source Image", "",
                "Images (*.png *.jpg *.jpeg *.exr *.tiff *.tif);;All Files (*)"
            )
            if path:
                self.source_image = self._load_image(path)
                if self.source_image is not None:
                    self.source_viewport.set_image(self.source_image)
                    self.status_bar.showMessage(f"Loaded: {Path(path).name}")
        
        @Slot()
        def load_reference(self):
            """Load reference image."""
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Reference Image", "",
                "Images (*.png *.jpg *.jpeg *.exr *.tiff *.tif);;All Files (*)"
            )
            if path:
                self.reference_image = self._load_image(path)
                if self.reference_image is not None:
                    self.reference_viewport.set_image(self.reference_image)
                    self.status_bar.showMessage(f"Reference loaded: {Path(path).name}")
        
        def _load_image(self, path: str) -> Optional[np.ndarray]:
            """Load image file."""
            try:
                import cv2
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img.astype(np.float32) / 255.0
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load image: {e}")
            return None
        
        @Slot()
        def match_colors(self):
            """Start color matching."""
            if self.source_image is None or self.reference_image is None:
                QMessageBox.warning(self, "Missing Images",
                                   "Please load both source and reference images.")
                return
            
            mode = self.mode_combo.currentText().lower()
            
            progress = QProgressDialog("Matching colors...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            self.worker = MatchWorker(self.source_image, self.reference_image, mode)
            self.worker.progress.connect(progress.setValue)
            self.worker.finished.connect(self._on_match_complete)
            self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
            self.worker.start()
        
        @Slot(dict)
        def _on_match_complete(self, result: dict):
            """Handle match completion."""
            self.match_result = result
            self.result_image = result['output']
            self.result_viewport.set_image(self.result_image)
            
            metrics = result.get('metrics', {})
            delta_e = metrics.get('delta_e_mean', 0)
            self.status_bar.showMessage(f"Match complete - ΔE00: {delta_e:.2f}")
            
            self.match_completed.emit(result)
        
        @Slot(str)
        def export(self, format: str):
            """Export matching result."""
            if self.match_result is None:
                QMessageBox.warning(self, "No Match", "Please match colors first.")
                return
            
            if format == "lut":
                path, _ = QFileDialog.getSaveFileName(
                    self, "Export 3D LUT", "match.cube", "Cube LUT (*.cube)"
                )
                if path:
                    from src.io.export import LUTExporter
                    exporter = LUTExporter()
                    exporter.export_cube(path, self.match_result['parameters'])
                    self.status_bar.showMessage(f"Exported: {path}")
        
        @Slot()
        def show_about(self):
            """Show about dialog."""
            QMessageBox.about(
                self, "About Chromatica Pro",
                "Chromatica Pro v1.0\n\n"
                "Professional Color Matching Software\n\n"
                "Features:\n"
                "- ACES 2.2 Pipeline\n"
                "- Parametric Matching (ΔE00 < 1.0)\n"
                "- Neural Vision Transformer\n"
                "- GPU Acceleration"
            )


def main():
    """Application entry point."""
    if not HAS_QT:
        print("ERROR: PySide6 not installed. Install with: pip install PySide6")
        return 1
    
    logging.basicConfig(level=logging.INFO)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Chromatica Pro")
    app.setApplicationVersion("1.0.0")
    
    window = ChromaticaMainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
