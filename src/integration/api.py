"""
Chromatica Pro REST API

HTTP API for studio integration and remote processing.
"""

import json
import logging
from typing import Dict, Optional
from pathlib import Path
import base64
import io

logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify, send_file
    HAS_FLASK = False  # Disabled by default - optional dependency
except ImportError:
    HAS_FLASK = False

import numpy as np


class ChromaticaAPI:
    """
    REST API for studio integration.
    
    Endpoints:
    - POST /match - Match source to reference
    - POST /batch - Batch match multiple images
    - POST /export/lut - Export 3D LUT
    - POST /export/cdl - Export CDL
    - GET /status - Server status
    - GET /cameras - List supported cameras
    
    Example:
        >>> api = ChromaticaAPI()
        >>> api.run(host='0.0.0.0', port=8080)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize engines
        from ..engines.hybrid.orchestrator import HybridOrchestrator
        self.orchestrator = HybridOrchestrator()
        
        from ..io.formats import ImageIO
        self.image_io = ImageIO
        
        logger.info("Chromatica API initialized")
    
    def match(self, source_path: str, reference_path: str,
              strategy: str = 'auto',
              output_path: Optional[str] = None) -> Dict:
        """
        Match source image to reference.
        
        Args:
            source_path: Path to source image
            reference_path: Path to reference image
            strategy: Matching strategy ('parametric', 'neural', 'hybrid', 'auto')
            output_path: Optional output path
            
        Returns:
            Match result with metrics and output path
        """
        # Load images
        source, _ = self.image_io.read(source_path)
        reference, _ = self.image_io.read(reference_path)
        
        # Match
        result = self.orchestrator.match(source, reference, strategy=strategy)
        
        # Save output
        if output_path:
            self.image_io.write(output_path, result['output'])
        
        return {
            'success': True,
            'metrics': result['metrics'],
            'strategy_used': result.get('strategy_used', strategy),
            'execution_time': result.get('total_time', 0),
            'output_path': output_path
        }
    
    def batch_match(self, source_paths: list, reference_path: str,
                    output_dir: str, strategy: str = 'auto') -> Dict:
        """
        Batch match multiple source images.
        
        Args:
            source_paths: List of source image paths
            reference_path: Path to reference image
            output_dir: Output directory
            strategy: Matching strategy
            
        Returns:
            Batch results with per-file metrics
        """
        reference, _ = self.image_io.read(reference_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total_time = 0
        
        for source_path in source_paths:
            source_path = Path(source_path)
            
            try:
                source, _ = self.image_io.read(str(source_path))
                result = self.orchestrator.match(source, reference, strategy=strategy)
                
                output_path = output_dir / source_path.name
                self.image_io.write(str(output_path), result['output'])
                
                results.append({
                    'source': str(source_path),
                    'output': str(output_path),
                    'metrics': result['metrics'],
                    'success': True
                })
                total_time += result.get('total_time', 0)
                
            except Exception as e:
                results.append({
                    'source': str(source_path),
                    'error': str(e),
                    'success': False
                })
        
        return {
            'total_images': len(source_paths),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'total_time': total_time,
            'results': results
        }
    
    def export_lut(self, parameters_path: str, output_path: str,
                   size: int = 33, format: str = 'cube') -> Dict:
        """Export 3D LUT from parameters."""
        from ..engines.parametric.parameters import ParametricParameters
        from ..io.export import LUTExporter
        
        params = ParametricParameters.load(parameters_path)
        exporter = LUTExporter()
        
        if format == 'cube':
            exporter.export_cube(output_path, params, size)
        elif format == '3dl':
            exporter.export_3dl(output_path, params, size)
        
        return {
            'success': True,
            'output_path': output_path,
            'size': size,
            'format': format
        }
    
    def get_supported_cameras(self) -> list:
        """Get list of supported camera profiles."""
        from ..core.aces_pipeline import ACESPipeline
        pipeline = ACESPipeline()
        return list(pipeline.camera_profiles.keys())
    
    def health_check(self) -> Dict:
        """API health check."""
        import torch
        
        return {
            'status': 'healthy',
            'version': '1.0.0',
            'cuda_available': torch.cuda.is_available(),
            'supported_strategies': ['parametric', 'neural', 'hybrid', 'auto']
        }
    
    def create_flask_app(self):
        """Create Flask application for HTTP API."""
        if not HAS_FLASK:
            raise ImportError("Flask not installed. Install with: pip install flask")
        
        app = Flask(__name__)
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify(self.health_check())
        
        @app.route('/match', methods=['POST'])
        def match():
            data = request.json
            result = self.match(
                data['source'],
                data['reference'],
                data.get('strategy', 'auto'),
                data.get('output')
            )
            return jsonify(result)
        
        @app.route('/batch', methods=['POST'])
        def batch():
            data = request.json
            result = self.batch_match(
                data['sources'],
                data['reference'],
                data['output_dir'],
                data.get('strategy', 'auto')
            )
            return jsonify(result)
        
        @app.route('/cameras', methods=['GET'])
        def cameras():
            return jsonify({'cameras': self.get_supported_cameras()})
        
        return app
    
    def run_server(self, host: str = '127.0.0.1', port: int = 8080):
        """Run HTTP API server."""
        app = self.create_flask_app()
        logger.info(f"Starting API server on {host}:{port}")
        app.run(host=host, port=port, debug=False)
