"""
Vision Transformer for Professional Color Matching

ChromaticaViT architecture for neural color matching with:
- Pre-trained ViT backbone for feature extraction
- Style encoder for reference image analysis  
- Color decoder for parameter prediction
- AdaIN for style transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Check for timm (Vision Transformer library)
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.warning("timm not installed. Neural model will use fallback.")


class StyleEncoder(nn.Module):
    """Extract style features from reference image."""
    
    def __init__(self, embed_dim: int = 768, style_dim: int = 128):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.style_mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, style_dim),
            nn.Tanh()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, N, D] from ViT
        pooled = features.mean(dim=1)  # Global average over patches
        return self.style_mlp(pooled)


class ColorDecoder(nn.Module):
    """Decode transformation parameters from features and style."""
    
    def __init__(self, embed_dim: int = 768, style_dim: int = 128, 
                 output_dim: int = 160):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim + style_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, features: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)
        combined = torch.cat([pooled, style], dim=1)
        return self.decoder(combined)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization for style transfer."""
    
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True) + 1e-6
        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True) + 1e-6
        
        normalized = (content - content_mean) / content_std
        return normalized * style_std + style_mean


class ChromaticaViT(nn.Module):
    """
    Vision Transformer for professional color matching.
    
    Architecture:
    - ViT backbone extracts features from source and reference
    - Style encoder captures reference "look"
    - Color decoder predicts parametric transform
    - Optional AdaIN refinement
    
    Example:
        >>> model = ChromaticaViT()
        >>> output, meta = model(source_tensor, reference_tensor)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or self._default_config()
        
        embed_dim = self.config['embed_dim']
        
        # ViT backbone
        if HAS_TIMM:
            self.vit = timm.create_model(
                self.config['backbone'],
                pretrained=self.config['pretrained'],
                num_classes=0,  # Remove classification head
                img_size=self.config['img_size']
            )
            # Get actual embed_dim from model
            embed_dim = self.vit.embed_dim
        else:
            self.vit = self._create_fallback_backbone()
            embed_dim = 768
        
        # Style extraction
        self.style_encoder = StyleEncoder(embed_dim, self.config['style_dim'])
        
        # Color transformation decoder
        self.color_decoder = ColorDecoder(
            embed_dim, 
            self.config['style_dim'],
            self.config['output_dim']
        )
        
        # Reconstruction head (optional)
        self.use_reconstruction = self.config.get('use_reconstruction', True)
        if self.use_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.Sigmoid()
            )
        
        self.adain = AdaIN()
    
    def _default_config(self) -> Dict:
        return {
            'backbone': 'vit_base_patch16_224',
            'pretrained': True,
            'img_size': 224,
            'embed_dim': 768,
            'style_dim': 128,
            'output_dim': 160,
            'use_reconstruction': True
        }
    
    def _create_fallback_backbone(self) -> nn.Module:
        """Simple CNN backbone when timm not available."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, source: torch.Tensor,
                reference: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        
        Args:
            source: Source image [B, 3, H, W]
            reference: Reference image [B, 3, H, W]
            
        Returns:
            (parameters, metadata) tuple
        """
        # Extract features
        if HAS_TIMM:
            source_features = self.vit.forward_features(source)
            reference_features = self.vit.forward_features(reference)
        else:
            source_features = self.vit(source).unsqueeze(1)
            reference_features = self.vit(reference).unsqueeze(1)
        
        # Extract style from reference
        style_embedding = self.style_encoder(reference_features)
        
        # Generate transformation parameters
        transform_params = self.color_decoder(source_features, style_embedding)
        
        metadata = {
            'transform_params': transform_params,
            'style_embedding': style_embedding,
            'source_features': source_features,
            'reference_features': reference_features
        }
        
        return transform_params, metadata
    
    def predict_parameters(self, source: torch.Tensor,
                          reference: torch.Tensor) -> torch.Tensor:
        """Predict only the transformation parameters."""
        params, _ = self.forward(source, reference)
        return params


class DeltaE2000Loss(nn.Module):
    """Differentiable approximation of ΔE2000."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_rgb: torch.Tensor, 
                target_rgb: torch.Tensor) -> torch.Tensor:
        # Convert to LAB (simplified)
        pred_lab = self._rgb_to_lab(pred_rgb)
        target_lab = self._rgb_to_lab(target_rgb)
        
        # Approximate ΔE2000 with weighted Euclidean
        dL = pred_lab[:, 0] - target_lab[:, 0]
        da = pred_lab[:, 1] - target_lab[:, 1]
        db = pred_lab[:, 2] - target_lab[:, 2]
        
        loss = torch.sqrt((dL / 1.0)**2 + (da / 1.5)**2 + (db / 1.5)**2)
        return loss.mean()
    
    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Differentiable RGB to LAB conversion."""
        # Linearize sRGB
        linear = torch.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4
        )
        
        # RGB to XYZ matrix
        M = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=rgb.device, dtype=rgb.dtype)
        
        # [B, 3, H, W] -> [B, H, W, 3] for matmul
        linear = linear.permute(0, 2, 3, 1)
        xyz = torch.matmul(linear, M.T)
        
        # Normalize by D65
        xyz_n = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device)
        xyz = xyz / xyz_n
        
        # f function
        epsilon = 216 / 24389
        kappa = 24389 / 27
        f = torch.where(
            xyz > epsilon,
            xyz ** (1/3),
            (kappa * xyz + 16) / 116
        )
        
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        
        # [B, H, W, 3] -> [B, 3, H, W]
        lab = torch.stack([L, a, b], dim=-1).permute(0, 3, 1, 2)
        return lab
