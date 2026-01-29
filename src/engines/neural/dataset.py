"""
Dataset for Color Matching Training

Handles source/reference/target image triplets with augmentation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torchvision.transforms as T
    HAS_TV = True
except ImportError:
    HAS_TV = False


class ColorMatchingDataset(Dataset):
    """
    Dataset for color matching training.
    
    Expects directory structure:
    data/
        source/
            image001.jpg
        target/
            image001.jpg
        reference/  (optional, defaults to target)
            image001.jpg
    """
    
    def __init__(self, data_dir: str, 
                 split: str = 'train',
                 img_size: int = 224,
                 augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        self.source_dir = self.data_dir / 'source'
        self.target_dir = self.data_dir / 'target'
        self.reference_dir = self.data_dir / 'reference'
        
        # Get file list
        self.files = self._get_file_list()
        
        # Transforms
        self.transform = self._create_transforms()
        
        logger.info(f"Loaded {len(self.files)} samples for {split}")
    
    def _get_file_list(self) -> List[str]:
        """Get list of matching files."""
        if not self.source_dir.exists():
            logger.warning(f"Source directory not found: {self.source_dir}")
            return []
        
        source_files = set(f.name for f in self.source_dir.glob('*') 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.exr', '.tiff'])
        
        if self.target_dir.exists():
            target_files = set(f.name for f in self.target_dir.glob('*'))
            common = source_files & target_files
        else:
            common = source_files
        
        return sorted(list(common))
    
    def _create_transforms(self):
        if not HAS_TV:
            return None
        
        transforms = [
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
        ]
        
        if self.augment:
            transforms.insert(0, T.RandomHorizontalFlip())
        
        return T.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.files[idx]
        
        # Load images
        source = self._load_image(self.source_dir / filename)
        target = self._load_image(self.target_dir / filename)
        
        if self.reference_dir.exists():
            reference = self._load_image(self.reference_dir / filename)
        else:
            reference = target.clone()
        
        return {
            'source': source,
            'target': target,
            'reference': reference,
            'filename': filename
        }
    
    def _load_image(self, path: Path) -> torch.Tensor:
        if not HAS_PIL:
            # Return dummy tensor
            return torch.randn(3, self.img_size, self.img_size)
        
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(np.array(img)).float() / 255.0
                img = img.permute(2, 0, 1)
            return img
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return torch.randn(3, self.img_size, self.img_size)
