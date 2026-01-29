#!/bin/bash
# Chromatica Pro Build Script for Linux/macOS
# Creates distribution packages

set -e

echo "=========================================="
echo "Chromatica Pro Build Script"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller build

# Run tests
echo "Running tests..."
pytest tests/ -v --tb=short || echo "WARNING: Some tests failed"

# Build wheel
echo "Building Python wheel..."
python -m build

# Create standalone executable
echo "Building standalone executable..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    pyinstaller --onefile --windowed \
        --name ChromaticaPro \
        --add-data "src:src" \
        src/ui/main_window.py
else
    # Linux
    pyinstaller --onefile \
        --name chromatica-pro \
        --add-data "src:src" \
        src/ui/main_window.py
fi

echo "=========================================="
echo "Build complete!"
echo "Wheel: dist/*.whl"
echo "Executable: dist/ChromaticaPro"
echo "=========================================="
