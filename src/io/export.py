"""
Export Functions for 3D LUT, CDL, DCTL formats.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LUTExporter:
    """Export 3D LUTs in various formats."""
    
    def export_cube(self, path: str, params, size: int = 33):
        """
        Export .cube format (compatible with Resolve, Premiere, etc.)
        
        Args:
            path: Output file path
            params: ParametricParameters or callable transform
            size: LUT size (17, 33, or 65 recommended)
        """
        from src.engines.parametric.solver import ParametricSolver
        
        path = Path(path)
        
        # Generate LUT values
        x = np.linspace(0, 1, size)
        r, g, b = np.meshgrid(x, x, x, indexing='ij')
        identity = np.stack([r, g, b], axis=-1)
        
        # Apply transform to identity LUT
        solver = ParametricSolver()
        lut = solver.apply(identity.reshape(-1, 3), params).reshape(size, size, size, 3)
        
        # Write .cube file
        with open(path, 'w') as f:
            f.write(f"# Created by Chromatica Pro\n")
            f.write(f"# {datetime.now().isoformat()}\n")
            f.write(f"TITLE \"Chromatica Match\"\n")
            f.write(f"LUT_3D_SIZE {size}\n")
            f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n")
            f.write("\n")
            
            # Write values in BGR order (cube format convention)
            for b_idx in range(size):
                for g_idx in range(size):
                    for r_idx in range(size):
                        r_val = lut[r_idx, g_idx, b_idx, 0]
                        g_val = lut[r_idx, g_idx, b_idx, 1]
                        b_val = lut[r_idx, g_idx, b_idx, 2]
                        f.write(f"{r_val:.6f} {g_val:.6f} {b_val:.6f}\n")
        
        logger.info(f"Exported {size}³ LUT to {path}")
    
    def export_3dl(self, path: str, params, size: int = 33, bit_depth: int = 12):
        """Export .3dl format (legacy Lustre/Autodesk)."""
        from src.engines.parametric.solver import ParametricSolver
        
        path = Path(path)
        max_val = (2 ** bit_depth) - 1
        
        x = np.linspace(0, 1, size)
        r, g, b = np.meshgrid(x, x, x, indexing='ij')
        identity = np.stack([r, g, b], axis=-1)
        
        solver = ParametricSolver()
        lut = solver.apply(identity.reshape(-1, 3), params).reshape(size, size, size, 3)
        
        with open(path, 'w') as f:
            # Header
            mesh_values = ' '.join(str(int(i * max_val / (size-1))) for i in range(size))
            f.write(f"#Chromatica Pro 3DL\n")
            f.write(f"{mesh_values}\n")
            
            # Values
            for b_idx in range(size):
                for g_idx in range(size):
                    for r_idx in range(size):
                        r_val = int(np.clip(lut[r_idx, g_idx, b_idx, 0], 0, 1) * max_val)
                        g_val = int(np.clip(lut[r_idx, g_idx, b_idx, 1], 0, 1) * max_val)
                        b_val = int(np.clip(lut[r_idx, g_idx, b_idx, 2], 0, 1) * max_val)
                        f.write(f"{r_val} {g_val} {b_val}\n")
        
        logger.info(f"Exported .3dl to {path}")


class CDLExporter:
    """Export ASC CDL format."""
    
    def export_cdl(self, path: str, cdl_params):
        """
        Export .cdl XML format.
        
        Args:
            path: Output file path
            cdl_params: CDLParameters object
        """
        path = Path(path)
        
        slope = ' '.join(f'{v:.6f}' for v in cdl_params.slope)
        offset = ' '.join(f'{v:.6f}' for v in cdl_params.offset)
        power = ' '.join(f'{v:.6f}' for v in cdl_params.power)
        sat = f'{cdl_params.saturation:.6f}'
        
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<ColorDecisionList xmlns="urn:ASC:CDL:v1.01">
    <ColorDecision>
        <ColorCorrection id="chromatica_match">
            <SOPNode>
                <Slope>{slope}</Slope>
                <Offset>{offset}</Offset>
                <Power>{power}</Power>
            </SOPNode>
            <SatNode>
                <Saturation>{sat}</Saturation>
            </SatNode>
        </ColorCorrection>
    </ColorDecision>
</ColorDecisionList>
'''
        
        with open(path, 'w') as f:
            f.write(xml)
        
        logger.info(f"Exported CDL to {path}")
    
    def export_ccc(self, path: str, cdl_list: list):
        """Export Color Correction Collection (.ccc) with multiple CDLs."""
        path = Path(path)
        
        corrections = []
        for i, cdl in enumerate(cdl_list):
            slope = ' '.join(f'{v:.6f}' for v in cdl.slope)
            offset = ' '.join(f'{v:.6f}' for v in cdl.offset)
            power = ' '.join(f'{v:.6f}' for v in cdl.power)
            sat = f'{cdl.saturation:.6f}'
            
            corrections.append(f'''
        <ColorCorrection id="correction_{i}">
            <SOPNode>
                <Slope>{slope}</Slope>
                <Offset>{offset}</Offset>
                <Power>{power}</Power>
            </SOPNode>
            <SatNode>
                <Saturation>{sat}</Saturation>
            </SatNode>
        </ColorCorrection>''')
        
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<ColorCorrectionCollection xmlns="urn:ASC:CDL:v1.01">
    {"".join(corrections)}
</ColorCorrectionCollection>
'''
        
        with open(path, 'w') as f:
            f.write(xml)
        
        logger.info(f"Exported {len(cdl_list)} CDLs to {path}")
