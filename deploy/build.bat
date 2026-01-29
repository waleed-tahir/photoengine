@echo off
REM Chromatica Pro Build Script for Windows
REM Creates distribution packages

echo ==========================================
echo Chromatica Pro Build Script
echo ==========================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    exit /b 1
)

REM Create virtual environment if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and install
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

REM Run tests
echo Running tests...
pytest tests/ -v --tb=short
if errorlevel 1 (
    echo WARNING: Some tests failed
)

REM Build wheel
echo Building Python wheel...
pip install build
python -m build

REM Create standalone executable
echo Building standalone executable...
pyinstaller --onefile --windowed ^
    --name ChromaticaPro ^
    --icon assets/icon.ico ^
    --add-data "src;src" ^
    src/ui/main_window.py

echo ==========================================
echo Build complete!
echo Wheel: dist/*.whl
echo Executable: dist/ChromaticaPro.exe
echo ==========================================
