"""
Advanced learning components for fuzzy clustering algorithms.

This module contains adaptive learning rate schedulers, early stopping criteria,
and convergence detection utilities.
"""

import numpy as np


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler for neural network training
    """
    def __init__(self, initial_lr=1e-3, patience=5, factor=0.5, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')

    def step(self, current_loss):
        """Update learning rate based on current loss"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait = 0
            return True  # Learning rate was updated
        return False

    def get_lr(self):
        return self.current_lr


class EarlyStoppingCriterion:
    """
    Early stopping criterion to prevent overfitting
    """
    def __init__(self, patience=10, min_delta=1e-6, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.wait = 0
        self.best_loss = float('inf')
        self.best_params = None

    def should_stop(self, current_loss, current_params=None):
        """Check if training should stop"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            if current_params is not None and self.restore_best:
                self.best_params = current_params.copy() if hasattr(current_params, 'copy') else current_params
        else:
            self.wait += 1

        return self.wait >= self.patience

    def get_best_params(self):
        return self.best_params


def adaptive_convergence_detection(history, window_size=5, threshold=1e-6):
    """
    Adaptive convergence detection based on loss history

    Args:
        history: List of loss values
        window_size: Size of the sliding window
        threshold: Convergence threshold
    """
    if len(history) < window_size + 1:
        return False

    # Check if the improvement in the last window_size iterations is below threshold
    recent_losses = history[-window_size-1:]
    improvements = [recent_losses[i] - recent_losses[i+1] for i in range(window_size)]
    avg_improvement = np.mean(improvements)

    return avg_improvement < threshold
