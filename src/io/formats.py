"""
Image I/O with format support and metadata preservation.
"""

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageIO:
    """
    Professional image I/O with format support.
    
    Supported formats:
    - Standard: PNG, JPEG, TIFF
    - Professional: EXR, DPX (with OpenImageIO)
    """
    
    SUPPORTED_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.tif', '.tiff',
        '.exr', '.dpx', '.hdr'
    }
    
    @staticmethod
    def read(path: str) -> Tuple[npt.NDArray, Dict]:
        """
        Read image file.
        
        Args:
            path: Image file path
            
        Returns:
            (image, metadata) tuple
            Image is float32 in 0-1 range (or scene-linear for EXR)
        """
        path = Path(path)
        ext = path.suffix.lower()
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        metadata = {'path': str(path), 'format': ext}
        
        if ext in ('.exr', '.hdr'):
            return ImageIO._read_hdr(path, metadata)
        elif HAS_CV2:
            return ImageIO._read_cv2(path, metadata)
        elif HAS_PIL:
            return ImageIO._read_pil(path, metadata)
        else:
            raise ImportError("No image library available")
    
    @staticmethod
    def _read_cv2(path: Path, metadata: Dict) -> Tuple[npt.NDArray, Dict]:
        """Read with OpenCV."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Failed to read: {path}")
        
        # Handle color conversion
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        
        metadata['shape'] = img.shape
        metadata['dtype'] = str(img.dtype)
        
        return img, metadata
    
    @staticmethod
    def _read_pil(path: Path, metadata: Dict) -> Tuple[npt.NDArray, Dict]:
        """Read with PIL."""
        img = Image.open(path)
        img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        
        metadata['shape'] = arr.shape
        metadata['mode'] = img.mode
        
        return arr, metadata
    
    @staticmethod
    def _read_hdr(path: Path, metadata: Dict) -> Tuple[npt.NDArray, Dict]:
        """Read HDR/EXR formats."""
        if HAS_CV2:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
            if img is not None and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            metadata['hdr'] = True
            return img.astype(np.float32), metadata
        raise ImportError("OpenCV required for HDR formats")
    
    @staticmethod
    def write(path: str, image: npt.NDArray, 
              bit_depth: int = 8,
              metadata: Optional[Dict] = None):
        """
        Write image file.
        
        Args:
            path: Output path
            image: Image array (0-1 range for LDR, any for HDR)
            bit_depth: Output bit depth (8, 16, 32)
            metadata: Optional metadata to preserve
        """
        path = Path(path)
        ext = path.suffix.lower()
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if ext in ('.exr', '.hdr'):
            ImageIO._write_hdr(path, image)
        elif HAS_CV2:
            ImageIO._write_cv2(path, image, bit_depth)
        elif HAS_PIL:
            ImageIO._write_pil(path, image)
        else:
            raise ImportError("No image library available")
    
    @staticmethod
    def _write_cv2(path: Path, image: npt.NDArray, bit_depth: int):
        """Write with OpenCV."""
        # Convert to BGR
        if image.ndim == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img = image
        
        # Scale and convert dtype
        if bit_depth == 8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == 16:
            img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        else:
            img = img.astype(np.float32)
        
        cv2.imwrite(str(path), img)
    
    @staticmethod
    def _write_pil(path: Path, image: npt.NDArray):
        """Write with PIL."""
        img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
    
    @staticmethod
    def _write_hdr(path: Path, image: npt.NDArray):
        """Write HDR formats."""
        if HAS_CV2:
            img = image.astype(np.float32)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), img)
        else:
            raise ImportError("OpenCV required for HDR formats")
