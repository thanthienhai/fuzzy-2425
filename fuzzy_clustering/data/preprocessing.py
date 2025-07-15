"""
Data preprocessing functions for different clustering algorithms.

This module contains preprocessing functions that prepare data
for specific clustering algorithms like DEKM and PFCM.
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess_usps_for_dekm(X_flat):
    """
    Preprocess USPS data for Deep Embedded K-Means (DEKM) algorithm.
    
    Args:
        X_flat: Flattened USPS image data
        
    Returns:
        tuple: (X_tensor, X_scaled) where X_tensor is PyTorch tensor 
               and X_scaled is numpy array of scaled data
    """
    if X_flat is None: 
        return None, None
    
    X_proc = X_flat.astype('float32') / 255.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)
    return torch.tensor(X_scaled, dtype=torch.float32), X_scaled


def preprocess_usps_for_pfcm(X_flat):
    """
    Preprocess USPS data for Possibilistic Fuzzy C-Means (PFCM) algorithm.
    
    Args:
        X_flat: Flattened USPS image data
        
    Returns:
        numpy.ndarray: PCA-reduced data or None if preprocessing fails
    """
    if X_flat is None: 
        return None
    
    X_proc = X_flat.astype('float32') / 255.0
    # Ensure n_components is not greater than number of samples or features
    n_components = min(50, X_proc.shape[0], X_proc.shape[1])
    if n_components < 1:
        print(f"Warning: Not enough features/samples for PCA ({X_proc.shape}). Returning raw scaled data for PFCM.")
        return X_proc # Or handle as an error
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_proc)
    return X_pca
