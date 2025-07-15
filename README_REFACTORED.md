# Fuzzy Clustering Algorithms - Refactored

This project has been refactored from a monolithic `run.py` file into a well-organized, modular Python package structure. The refactoring improves code maintainability, reusability, and follows Python best practices.

## Project Structure

```
fuzzy-2425/
├── fuzzy_clustering/           # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── utils/                 # Utility modules
│   │   ├── __init__.py
│   │   ├── performance.py     # Performance optimization utilities
│   │   ├── learning.py        # Adaptive learning components
│   │   └── initialization.py  # Smart initialization strategies
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loaders.py         # Dataset loading functions
│   │   └── preprocessing.py   # Data preprocessing functions
│   ├── models/                # Neural network models
│   │   ├── __init__.py
│   │   └── autoencoder.py     # Autoencoder architectures
│   ├── algorithms/            # Clustering algorithms
│   │   ├── __init__.py
│   │   ├── dekm.py           # Deep Embedded K-Means
│   │   ├── pfcm.py           # Possibilistic Fuzzy C-Means
│   │   ├── fcm.py            # Fuzzy C-Means
│   │   ├── kmeans.py         # K-Means
│   │   ├── dbscan.py         # DBSCAN
│   │   ├── cfcm.py           # Collaborative Fuzzy C-Means
│   │   └── fdekm.py          # Fuzzy Deep Embedded K-Means
│   ├── metrics/               # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── sklearn_metrics.py # Standard sklearn metrics
│   │   └── fuzzy_metrics.py   # Custom fuzzy clustering metrics
│   └── main.py               # Main execution logic
├── main.py                   # Entry point script
├── run.py                    # Original monolithic file (preserved)
├── requirements.txt          # Python dependencies
└── README_REFACTORED.md      # This documentation
```

## Key Improvements

### 1. **Separation of Concerns**
- **Data handling**: Isolated in `data/` module
- **Algorithms**: Each algorithm in its own file in `algorithms/`
- **Metrics**: Evaluation metrics separated in `metrics/`
- **Utilities**: Common utilities in `utils/`
- **Models**: Neural network models in `models/`

### 2. **Modularity**
- Each module has a single responsibility
- Easy to import and use individual components
- Facilitates testing and debugging

### 3. **Maintainability**
- Clear module boundaries
- Consistent naming conventions
- Comprehensive docstrings
- Error handling improvements

### 4. **Reusability**
- Algorithms can be used independently
- Data loaders can be reused for different experiments
- Metrics can be applied to any clustering result

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run all algorithms on all datasets
python main.py
```

### Using Individual Components
```python
from fuzzy_clustering.algorithms import run_fcm, run_dekm
from fuzzy_clustering.data import load_usps_data, preprocess_usps_for_dekm
from fuzzy_clustering.metrics import calculate_sklearn_metrics

# Load and preprocess data
X_flat, y = load_usps_data()
X_tensor, X_scaled = preprocess_usps_for_dekm(X_flat)

# Run specific algorithm
labels, embeddings, metrics = run_dekm(X_tensor, X_scaled, k=10, verbose=True)

# Evaluate results
evaluation_metrics = calculate_sklearn_metrics(X_scaled, labels)
```

## Algorithms Implemented

1. **Deep Embedded K-Means (DEKM)**: Combines autoencoder with K-means clustering
2. **Possibilistic Fuzzy C-Means (PFCM)**: Fuzzy clustering with possibilistic approach
3. **Fuzzy C-Means (FCM)**: Standard fuzzy clustering algorithm
4. **Collaborative Fuzzy C-Means (CFCM)**: FCM with feature collaboration
5. **Fuzzy Deep Embedded K-Means (FDEKM)**: Fuzzy version of DEKM
6. **K-Means**: Traditional hard clustering
7. **DBSCAN**: Density-based clustering

## Datasets Supported

1. **USPS Handwritten Digits**: 16x16 grayscale images of digits 0-9
2. **E-commerce Data**: Customer transaction data for segmentation
3. **Country Development Data**: Socio-economic indicators for country clustering

## Evaluation Metrics

### Standard Metrics (sklearn)
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

### Fuzzy Clustering Metrics
- Partition Coefficient Index (PCI)
- Fuzzy Hypervolume (FHV)
- Xie-Beni Index (XBI)

## Migration from Original Code

The original `run.py` file (2,253 lines) has been refactored into:
- **8 algorithm modules** (~50-300 lines each)
- **6 utility modules** (~50-150 lines each)
- **3 data modules** (~50-100 lines each)
- **3 metrics modules** (~50-150 lines each)
- **2 model modules** (~100-200 lines each)
- **1 main execution module** (~300 lines)

### Benefits of Refactoring
- **Reduced complexity**: Each file focuses on a specific functionality
- **Improved testability**: Individual components can be tested in isolation
- **Better collaboration**: Multiple developers can work on different modules
- **Enhanced documentation**: Each module has clear purpose and documentation
- **Easier debugging**: Issues can be isolated to specific modules

## Future Enhancements

1. **Complete CFCM and FDEKM implementations**: Currently simplified versions
2. **Add unit tests**: Comprehensive test suite for all modules
3. **Performance optimization**: Further optimize algorithms for large datasets
4. **Visualization tools**: Add plotting and visualization capabilities
5. **Configuration management**: Add configuration files for algorithm parameters
6. **Parallel processing**: Add support for multi-processing and GPU acceleration

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key requirements:
- Python 3.7+
- NumPy, Pandas, SciPy
- Scikit-learn, Scikit-fuzzy
- PyTorch
- H5PY

## Contributing

When contributing to this refactored codebase:
1. Follow the established module structure
2. Add comprehensive docstrings
3. Include error handling
4. Update this README if adding new modules
5. Ensure backward compatibility with the original functionality
