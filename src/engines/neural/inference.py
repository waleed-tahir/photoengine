"""
Neural Inference Engine

Production-ready inference with batching and optimization.
"""

import torch
import numpy as np
import numpy.typing as npt
from typing import Dict, Optional
import logging

from .model import ChromaticaViT
from ..parametric.parameters import ParametricParameters

logger = logging.getLogger(__name__)


class NeuralInference:
    """
    Production inference engine for neural color matching.
    
    Example:
        >>> engine = NeuralInference('model.pt')
        >>> params = engine.predict(source_image, reference_image)
        >>> matched = engine.apply(source_image, params)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = ChromaticaViT().to(self.device)
        self.model.eval()
        
        if model_path:
            self.load_weights(model_path)
        
        logger.info(f"Neural inference engine ready on {self.device}")
    
    def load_weights(self, path: str):
        """Load pre-trained weights."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from {path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
    
    def predict(self, source: npt.NDArray, 
                reference: npt.NDArray) -> ParametricParameters:
        """
        Predict transformation parameters.
        
        Args:
            source: Source image [H, W, 3] in 0-1 range
            reference: Reference image [H, W, 3] in 0-1 range
            
        Returns:
            ParametricParameters predicted by model
        """
        # Prepare tensors
        source_t = self._prepare_tensor(source)
        reference_t = self._prepare_tensor(reference)
        
        with torch.no_grad():
            params_t, _ = self.model(source_t, reference_t)
        
        # Convert to ParametricParameters
        params_np = params_t.cpu().numpy().squeeze()
        
        # Scale from [-1, 1] tanh output to parameter ranges
        params_scaled = self._scale_parameters(params_np)
        
        return ParametricParameters.from_vector(params_scaled)
    
    def _prepare_tensor(self, image: npt.NDArray) -> torch.Tensor:
        """Convert numpy image to model input tensor."""
        # Ensure float
        image = np.asarray(image, dtype=np.float32)
        
        # Resize to model input size
        import cv2
        image = cv2.resize(image, (224, 224))
        
        # [H, W, C] -> [1, C, H, W]
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _scale_parameters(self, params: npt.NDArray) -> npt.NDArray:
        """Scale tanh output [-1, 1] to parameter ranges."""
        from ..parametric.parameters import get_parameter_bounds
        lower, upper = get_parameter_bounds()
        
        # Scale from [-1, 1] to [lower, upper]
        params_scaled = lower + (params + 1) / 2 * (upper - lower)
        return params_scaled
    
    def apply(self, image: npt.NDArray, 
              params: ParametricParameters) -> npt.NDArray:
        """Apply predicted parameters to image."""
        from ..parametric.solver import ParametricSolver
        solver = ParametricSolver()
        return solver.apply(image, params)
