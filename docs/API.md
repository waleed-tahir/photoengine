# Chromatica Pro API Documentation

## Overview

Chromatica Pro is a professional color matching application for film and television workflows.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### CLI Usage

```bash
# Match a single image
chromatica match source.jpg reference.jpg -o output.jpg

# Export as 3D LUT
chromatica match source.jpg reference.jpg --export-lut match.cube

# Batch processing
chromatica batch ./input/ ./output/ reference.jpg -m parametric
```

### Python API

```python
from src.engines.hybrid.orchestrator import HybridOrchestrator
from src.io.formats import ImageIO

# Load images
source, _ = ImageIO.read("source.jpg")
reference, _ = ImageIO.read("reference.jpg")

# Match
orchestrator = HybridOrchestrator()
result = orchestrator.match(source, reference, strategy='hybrid')

# Access results
print(f"ΔE00: {result['metrics']['delta_e_mean']:.2f}")
matched_image = result['output']
parameters = result['parameters']
```

---

## Core Modules

### ACES Pipeline

```python
from src.core.aces_pipeline import ACESPipeline, LMTParams, CDLParams

pipeline = ACESPipeline()

# Apply IDT (Input Device Transform)
aces_image = pipeline.apply_idt(camera_image, 'arri_logc4')

# Apply LMT (Look Modification Transform)
lmt = LMTParams(cdl=CDLParams(slope=(1.1, 1.0, 0.9)))
graded = pipeline.apply_lmt(aces_image, lmt)

# Apply RRT+ODT for display
display_image = pipeline.apply_rrt_odt(graded, 'srgb')
```

**Supported Cameras:**
- ARRI (LogC3, LogC4)
- Sony (S-Log3)
- RED (Log3G10)
- Canon (C-Log3)
- Blackmagic (BMD Film Gen5)
- Panasonic (V-Log)
- DJI (D-Log)

### Color Space Converter

```python
from src.core.color_spaces import ColorSpaceConverter

converter = ColorSpaceConverter()

# RGB to LAB
lab = converter.rgb_to_lab(rgb_image)

# RGB to XYZ with custom color space
xyz = converter.rgb_to_xyz(rgb_image, source_space='rec2020')

# Chromatic adaptation
adapted = converter.chromatic_adaptation(xyz, 'd65', 'd50')
```

### Transforms

```python
from src.core.transforms import ColorTransforms, LUT3D

# Apply CDL
result = ColorTransforms.apply_cdl(
    image,
    slope=(1.1, 1.0, 0.95),
    offset=(0.01, 0.0, -0.01),
    power=(1.0, 1.0, 1.0),
    saturation=1.1
)

# Apply 3D LUT
lut = LUT3D.load("my_lut.cube")
result = lut.apply(image, method='tetrahedral')
```

---

## Matching Engines

### Parametric Solver

160-parameter optimization for professional color matching:

```python
from src.engines.parametric.solver import ParametricSolver
from src.engines.parametric.parameters import ParametricParameters

solver = ParametricSolver(config={
    'max_iterations': 500,
    'tolerance': 1e-7
})

result = solver.match(source, target)
params = result['parameters']  # ParametricParameters object

# Apply to new images
transformed = solver.apply(new_image, params)

# Save parameters
params.save("match_params.json")
```

### Neural Engine

Vision Transformer for complex look matching:

```python
from src.engines.neural.inference import NeuralInference

engine = NeuralInference("model_weights.pt")
params = engine.predict(source, reference)
result = engine.apply(source, params)
```

### Hybrid Orchestrator

Intelligent engine selection:

```python
from src.engines.hybrid.orchestrator import HybridOrchestrator

orchestrator = HybridOrchestrator()

# Auto-select best strategy
result = orchestrator.match(source, reference, strategy='auto')

# Force specific strategy
result = orchestrator.match(source, reference, strategy='neural')
```

---

## Export Formats

### 3D LUT (.cube)

```python
from src.io.export import LUTExporter

exporter = LUTExporter()
exporter.export_cube("output.cube", parameters, size=33)
```

### ASC CDL

```python
from src.io.export import CDLExporter

exporter = CDLExporter()
exporter.export_cdl("output.cdl", cdl_params)
```

---

## Validation

### ColorChecker Validation

```python
from src.analysis.validation import ColorCheckerValidator

validator = ColorCheckerValidator()
result = validator.validate(matched_lab_patches)

print(f"Mean ΔE00: {result.mean_delta_e:.2f}")
print(f"Patches under 1.0: {result.patches_under_1}/24")
print(validator.generate_report(result))
```

### Quality Analysis

```python
from src.analysis.quality import QualityAnalyzer

analyzer = QualityAnalyzer()
report = analyzer.analyze(result_image, target_image)

print(f"Grade: {report.overall_grade}")
print(f"PSNR: {report.psnr:.2f} dB")
```

---

## GPU Acceleration

```python
from src.gpu.acceleration import GPUAccelerator

gpu = GPUAccelerator(backend='auto')  # Selects CuPy/CUDA or CPU

# GPU-accelerated LUT
result = gpu.apply_3d_lut(image, lut)

# GPU matrix transform
result = gpu.apply_matrix(image, matrix_3x4)
```

---

## REST API

Start API server:

```bash
python -m src.integration.api
```

Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /match | Match single image |
| POST | /batch | Batch match |
| GET | /cameras | List supported cameras |

Example request:
```bash
curl -X POST http://localhost:8080/match \
  -H "Content-Type: application/json" \
  -d '{"source": "/path/source.jpg", "reference": "/path/ref.jpg"}'
```
