#!/usr/bin/env python3
"""
Main entry point for the fuzzy clustering algorithms project.

This script provides a clean interface to run the refactored fuzzy clustering
algorithms on various datasets. It replaces the original monolithic run.py
with a modular, well-organized structure.

Usage:
    python main.py

The script will:
1. Load datasets (USPS, e-commerce, country data)
2. Run various clustering algorithms (DEKM, PFCM, FCM, K-Means, DBSCAN, etc.)
3. Evaluate performance using multiple metrics
4. Save results to CSV file

Requirements:
    - All dependencies from the original run.py
    - The fuzzy_clustering package modules
"""

import sys
import os

# Add the current directory to Python path to import fuzzy_clustering package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fuzzy_clustering.main import main
    
    if __name__ == '__main__':
        print("="*60)
        print("FUZZY CLUSTERING ALGORITHMS - REFACTORED VERSION")
        print("="*60)
        print("Running comprehensive clustering evaluation...")
        print("This may take several minutes depending on your hardware.")
        print("="*60)
        
        main()
        
except ImportError as e:
    print(f"Error importing fuzzy_clustering package: {e}")
    print("Please ensure all required dependencies are installed:")
    print("- pandas")
    print("- numpy") 
    print("- h5py")
    print("- scikit-learn")
    print("- torch")
    print("- scikit-fuzzy")
    print("- scipy")
    sys.exit(1)
except Exception as e:
    print(f"Error running clustering algorithms: {e}")
    sys.exit(1)
