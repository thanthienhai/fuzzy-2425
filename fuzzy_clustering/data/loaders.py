"""
Data loading functions for various datasets.

This module contains functions to load USPS handwritten digits,
e-commerce data, and country development data.
"""

import os
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler


def load_usps_data(base_path='.'):
    """
    Load USPS handwritten digits dataset from HDF5 file.
    
    Args:
        base_path: Base directory path where usps.h5 is located
        
    Returns:
        tuple: (X_flat, y) where X_flat is flattened image data and y is labels
    """
    file_path = os.path.join(base_path, 'usps.h5')
    try:
        with h5py.File(file_path, 'r') as f:
            train_X = np.array(f['train']['data'])
            train_y = np.array(f['train']['target'])
            test_X = np.array(f['test']['data'])
            test_y = np.array(f['test']['target'])
        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        X_flat = X.reshape((X.shape[0], -1))
        print(f"USPS data loaded successfully from {file_path}. Shape: {X_flat.shape}")
        return X_flat, y
    except FileNotFoundError:
        print(f"Error: USPS data file '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading USPS data from '{file_path}': {e}")
        return None, None


def load_ecommerce_data(base_path='.'):
    """
    Load e-commerce dataset from CSV files.
    
    Args:
        base_path: Base directory path where FCM_NA folder is located
        
    Returns:
        numpy.ndarray: Scaled feature matrix or None if loading fails
    """
    orders_path = os.path.join(base_path, 'FCM_NA', 'List of Orders.csv')
    details_path = os.path.join(base_path, 'FCM_NA', 'Order Details.csv')
    try:
        orders = pd.read_csv(orders_path)
        order_details = pd.read_csv(details_path)
        print(f"E-commerce data loaded from {orders_path} and {details_path}")
    except FileNotFoundError:
        print(f"Error: E-commerce CSV files not found. Checked: '{orders_path}' and '{details_path}'.")
        return None
    except Exception as e:
        print(f"Error loading E-commerce data: {e}")
        return None
        
    orders.dropna(inplace=True)
    order_details.dropna(inplace=True)
    df = pd.merge(order_details, orders, on='Order ID')
    features = ['Amount', 'Profit', 'Quantity']
    if not all(feature in df.columns for feature in features):
        print(f"Error: Required features {features} not all found in merged e-commerce data.")
        return None
    data = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    return X_scaled


def load_country_data(base_path='.'):
    """
    Load country development data from CSV file.
    
    Args:
        base_path: Base directory path where Country-data.csv is located
        
    Returns:
        tuple: (X_scaled, country_names) where X_scaled is scaled features 
               and country_names is the country column
    """
    file_path = os.path.join(base_path, 'Country-data.csv')
    try:
        df = pd.read_csv(file_path)
        print(f"Country data loaded successfully from {file_path}. Shape: {df.shape}")
        
        # Drop the 'country' column as it's not a feature
        features = df.drop('country', axis=1)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        return X_scaled, df['country']
    except FileNotFoundError:
        print(f"Error: Country data file '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading Country data from '{file_path}': {e}")
        return None, None
