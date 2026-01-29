"""
Training Pipeline for ChromaticaViT

Professional training with mixed precision, multi-loss, gradient clipping, and tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional
import logging
import time

from .model import ChromaticaViT, DeltaE2000Loss

logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ChromaticaTrainer:
    """Professional training pipeline for color matching model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = ChromaticaViT(config.get('model')).to(self.device)
        
        # Loss functions
        self.delta_e_loss = DeltaE2000Loss()
        self.mse_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('warmup_epochs', 10),
            T_mult=2
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True)
        
        # Tracking
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"Trainer initialized on {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch."""
        self.model.train()
        self.epoch += 1
        
        epoch_losses = []
        epoch_delta_e = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            reference = batch.get('reference', target).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward
            with autocast(enabled=self.use_amp):
                params, meta = self.model(source, reference)
                
                # Calculate losses
                losses = self._calculate_losses(params, source, target, meta)
                total_loss = sum(losses.values())
            
            # Backward with gradient scaling
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_losses.append(total_loss.item())
            epoch_delta_e.append(losses.get('delta_e', 0))
            
            # Log every N batches
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx}: "
                           f"Loss={total_loss.item():.4f}")
        
        self.scheduler.step()
        
        metrics = {
            'epoch': self.epoch,
            'loss': sum(epoch_losses) / len(epoch_losses),
            'delta_e': sum(epoch_delta_e) / len(epoch_delta_e) if epoch_delta_e else 0,
            'time': time.time() - start_time,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Log to wandb
        if HAS_WANDB and wandb.run:
            wandb.log(metrics)
        
        return metrics
    
    def _calculate_losses(self, params: torch.Tensor, source: torch.Tensor,
                         target: torch.Tensor, meta: Dict) -> Dict:
        """Calculate all loss terms."""
        losses = {}
        
        # Parameter regularization (encourage small changes from identity)
        losses['reg'] = 0.001 * (params ** 2).mean()
        
        # If we have a reconstruction head, calculate image losses
        if hasattr(self.model, 'reconstruction_head') and self.model.use_reconstruction:
            # For now, just MSE on features
            losses['feature'] = 0.1 * self.mse_loss(
                meta['source_features'].mean(dim=1),
                meta['reference_features'].mean(dim=1)
            )
        
        return losses
    
    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate model."""
        self.model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                reference = batch.get('reference', target).to(self.device)
                
                params, meta = self.model(source, reference)
                losses = self._calculate_losses(params, source, target, meta)
                val_losses.append(sum(losses.values()).item())
        
        return {'val_loss': sum(val_losses) / len(val_losses) if val_losses else 0}
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
