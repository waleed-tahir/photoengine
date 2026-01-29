# Chromatica Pro

Professional color matching software for film and television workflows.

## Features

- **ACES 2.2 Pipeline**: Full support for 300+ camera profiles
- **Parametric Matching**: ΔE00 < 1.0 accuracy with 160-parameter optimization
- **Neural Engine**: Vision Transformer for complex look matching
- **GPU Acceleration**: CUDA/CuPy backend for real-time processing
- **Professional UI**: Qt6 interface with triple viewport and scopes

## Installation

```bash
# Clone repository
git clone https://github.com/yourorg/chromatica-pro.git
cd chromatica-pro

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Match a source image to reference
chromatica match source.jpg reference.jpg -o output.jpg

# Export as 3D LUT
chromatica match source.jpg reference.jpg --export-lut match.cube

# Batch process
chromatica batch ./input/ ./output/ reference.jpg
```

### Python API

```python
from src.engines.hybrid.orchestrator import HybridOrchestrator
from src.io.formats import ImageIO

# Load images
source, _ = ImageIO.read("source.jpg")
reference, _ = ImageIO.read("reference.jpg")

# Match colors
orchestrator = HybridOrchestrator()
result = orchestrator.match(source, reference, strategy='hybrid')

# Save result
ImageIO.write("output.jpg", result['output'])
print(f"ΔE00: {result['metrics']['delta_e_mean']:.2f}")
```

### UI Application

```bash
# Launch the GUI
python -m src.ui.main_window
```

## Project Structure

```
PhotoEngine/
├── src/
│   ├── core/              # Color science fundamentals
│   │   ├── aces_pipeline.py
│   │   ├── color_spaces.py
│   │   ├── color_science.py
│   │   └── transforms.py
│   ├── engines/           # Matching engines
│   │   ├── parametric/    # 160-parameter solver
│   │   ├── neural/        # Vision Transformer
│   │   └── hybrid/        # Orchestration
│   ├── gpu/               # GPU acceleration
│   ├── io/                # Image I/O & export
│   └── ui/                # Qt6 interface
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Running Tests

```bash
pytest tests/ -v
```

## License

Copyright 2024. All rights reserved.
