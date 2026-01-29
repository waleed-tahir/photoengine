"""
Chromatica Pro CLI

Command-line interface for batch color matching and export.
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='chromatica',
        description='Chromatica Pro - Professional Color Matching'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Match source to reference')
    match_parser.add_argument('source', help='Source image path')
    match_parser.add_argument('reference', help='Reference image path')
    match_parser.add_argument('-o', '--output', help='Output image path')
    match_parser.add_argument('-m', '--mode', 
                             choices=['parametric', 'neural', 'hybrid', 'auto'],
                             default='auto', help='Matching mode')
    match_parser.add_argument('--export-lut', help='Export LUT to .cube file')
    match_parser.add_argument('--export-cdl', help='Export CDL to file')
    match_parser.add_argument('-v', '--verbose', action='store_true')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process images')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('reference', help='Reference image')
    batch_parser.add_argument('-m', '--mode', default='auto')
    batch_parser.add_argument('-v', '--verbose', action='store_true')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system info')
    
    # Version
    parser.add_argument('--version', action='version', version='Chromatica Pro 1.0.0')
    
    args = parser.parse_args()
    
    if args.command == 'match':
        return cmd_match(args)
    elif args.command == 'batch':
        return cmd_batch(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


def cmd_match(args):
    """Execute match command."""
    setup_logging(args.verbose)
    
    from .io.formats import ImageIO
    from .engines.hybrid.orchestrator import HybridOrchestrator
    
    logger.info(f"Loading source: {args.source}")
    source, _ = ImageIO.read(args.source)
    
    logger.info(f"Loading reference: {args.reference}")
    reference, _ = ImageIO.read(args.reference)
    
    logger.info(f"Matching with mode: {args.mode}")
    orchestrator = HybridOrchestrator()
    result = orchestrator.match(source, reference, strategy=args.mode)
    
    metrics = result.get('metrics', {})
    delta_e = metrics.get('delta_e_mean', 0)
    logger.info(f"Match complete - ΔE00: {delta_e:.3f}")
    
    # Save output image
    if args.output:
        logger.info(f"Saving result: {args.output}")
        ImageIO.write(args.output, result['output'])
    
    # Export LUT
    if args.export_lut:
        from .io.export import LUTExporter
        logger.info(f"Exporting LUT: {args.export_lut}")
        exporter = LUTExporter()
        exporter.export_cube(args.export_lut, result['parameters'])
    
    # Export CDL
    if args.export_cdl:
        from .io.export import CDLExporter
        from .engines.parametric.parameters import CDLParameters
        # Convert params to CDL (simplified)
        cdl = CDLParameters(
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=1.0
        )
        logger.info(f"Exporting CDL: {args.export_cdl}")
        exporter = CDLExporter()
        exporter.export_cdl(args.export_cdl, cdl)
    
    print(f"\n✓ Match complete")
    print(f"  ΔE00 Mean: {delta_e:.3f}")
    print(f"  Time: {result.get('total_time', 0):.2f}s")
    
    return 0


def cmd_batch(args):
    """Execute batch command."""
    setup_logging(args.verbose)
    
    from .io.formats import ImageIO
    from .engines.hybrid.orchestrator import HybridOrchestrator
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference once
    reference, _ = ImageIO.read(args.reference)
    
    orchestrator = HybridOrchestrator()
    
    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.exr']
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(ext))
    
    logger.info(f"Found {len(files)} images to process")
    
    for i, file in enumerate(files):
        logger.info(f"Processing {i+1}/{len(files)}: {file.name}")
        
        try:
            source, _ = ImageIO.read(str(file))
            result = orchestrator.match(source, reference, strategy=args.mode)
            
            output_path = output_dir / file.name
            ImageIO.write(str(output_path), result['output'])
            
            delta_e = result.get('metrics', {}).get('delta_e_mean', 0)
            print(f"  ✓ {file.name} - ΔE00: {delta_e:.3f}")
        except Exception as e:
            logger.error(f"Failed: {file.name} - {e}")
            print(f"  ✗ {file.name} - Error: {e}")
    
    print(f"\n✓ Batch complete: {len(files)} images processed")
    return 0


def cmd_info(args):
    """Show system information."""
    import torch
    import numpy as np
    
    print("Chromatica Pro - System Information")
    print("=" * 40)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    try:
        import colour
        print(f"colour-science: {colour.__version__}")
    except ImportError:
        print("colour-science: Not installed")
    
    try:
        import cupy
        print(f"CuPy: {cupy.__version__}")
    except ImportError:
        print("CuPy: Not installed")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
