"""
Autoencoder models for deep clustering algorithms.

This module contains autoencoder architectures and training functions
used in Deep Embedded K-Means (DEKM) and related algorithms.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ..utils.performance import timing_decorator
from ..utils.learning import AdaptiveLearningRateScheduler, EarlyStoppingCriterion, adaptive_convergence_detection


class AutoEncoder_DEKM(nn.Module):
    """
    Autoencoder architecture for Deep Embedded K-Means (DEKM) algorithm.
    
    Features adaptive architecture based on input dimensions, batch normalization,
    dropout for regularization, and bounded activations for stability.
    """
    
    def __init__(self, input_dim, hidden_dim_ae, dropout_rate=0.2):
        super(AutoEncoder_DEKM, self).__init__()

        # Ensure valid dimensions
        if hidden_dim_ae <= 0:
            hidden_dim_ae = max(1, input_dim // 4)
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive")

        # Adaptive intermediate dimension based on input size
        if input_dim <= 64:
            intermediate_dim = max(hidden_dim_ae * 2, 32)
        elif input_dim <= 256:
            intermediate_dim = max(hidden_dim_ae * 3, 128)
        elif input_dim <= 1024:
            intermediate_dim = max(hidden_dim_ae * 4, 256)
        else:
            intermediate_dim = max(hidden_dim_ae * 4, 512)

        # Ensure intermediate dimension is reasonable
        intermediate_dim = min(intermediate_dim, input_dim * 2)
        intermediate_dim = max(intermediate_dim, hidden_dim_ae)

        # Build encoder with batch normalization and dropout
        if input_dim <= hidden_dim_ae:
            # Simple case: direct mapping
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_ae),
                nn.BatchNorm1d(hidden_dim_ae),
                nn.Tanh()  # Bounded activation for better stability
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim_ae, input_dim),
                nn.Sigmoid()  # Assuming normalized input data
            )
        else:
            # Multi-layer encoder with proper regularization
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_dim, hidden_dim_ae),
                nn.BatchNorm1d(hidden_dim_ae),
                nn.Tanh()  # Bounded activation for latent space
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim_ae, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_dim, input_dim),
                nn.Sigmoid()  # Assuming normalized input data
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_ae

    def forward(self, x):
        """Forward pass through encoder and decoder"""
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def encode(self, x):
        """Get only the encoded representation"""
        return self.encoder(x)


@timing_decorator
def train_autoencoder_dekm(model, data, epochs=100, lr=1e-3, weight_decay=1e-5,
                          patience=10, min_delta=1e-6, verbose=False, use_adaptive_lr=True):
    """
    Enhanced autoencoder training with adaptive learning rates and advanced features.
    
    Args:
        model: AutoEncoder_DEKM model to train
        data: Training data tensor
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        weight_decay: L2 regularization weight
        patience: Patience for early stopping
        min_delta: Minimum change for early stopping
        verbose: Whether to print training progress
        use_adaptive_lr: Whether to use adaptive learning rate scheduling
    """
    if data.shape[0] == 0:
        print("Error: DEKM Autoencoder training data is empty.")
        return

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Initialize advanced features
    if use_adaptive_lr:
        lr_scheduler = AdaptiveLearningRateScheduler(initial_lr=lr, patience=patience//3)
    early_stopping = EarlyStoppingCriterion(patience=patience, min_delta=min_delta)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, latent = model(data)

        # Reconstruction loss
        recon_loss = criterion(output, data)

        # Add regularization terms
        l2_reg = torch.norm(latent, p=2, dim=1).mean()

        # Sparsity regularization on latent representations
        sparsity_reg = torch.mean(torch.abs(latent))

        # Total loss with adaptive weighting
        total_loss = recon_loss + 0.001 * l2_reg + 0.0001 * sparsity_reg

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update learning rate adaptively
        if use_adaptive_lr:
            lr_updated = lr_scheduler.step(total_loss.item())
            if lr_updated:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scheduler.get_lr()
                if verbose:
                    print(f"Learning rate updated to {lr_scheduler.get_lr():.6f}")

        # Track loss history
        loss_history.append(total_loss.item())

        # Check early stopping
        if early_stopping.should_stop(total_loss.item()):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

        # Adaptive convergence detection
        if epoch > 10 and adaptive_convergence_detection(loss_history, window_size=5):
            if verbose:
                print(f"Adaptive convergence detected at epoch {epoch+1}")
            break

        if verbose and (epoch+1) % 20 == 0:
            print(f"AE Epoch {epoch+1}/{epochs}, Loss: {recon_loss.item():.6f}, "
                  f"L2: {l2_reg.item():.6f}, Sparsity: {sparsity_reg.item():.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
