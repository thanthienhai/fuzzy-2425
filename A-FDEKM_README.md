# A-FDEKM: Attentive Fuzzy Deep Embedded K-Means

## Overview

A-FDEKM (Attentive Fuzzy Deep Embedded K-Means) is an enhanced version of the FDEKM algorithm designed to address initialization sensitivity, "black box" nature, and cluster separability issues. It integrates three main improvements over the original FDEKM algorithm.

## Key Improvements

### 1. Attention Mechanism
- **Problem**: Traditional FDEKM treats all features (L, R, F, M indices) equally
- **Solution**: Integrates an attention layer that automatically learns feature importance
- **Benefit**: Creates context-aware representation space and enhances explainability

### 2. Dual Loss Function
- **Problem**: Original FDEKM only minimizes intra-cluster compactness
- **Solution**: Enhanced loss function with two objectives:
  - Minimize intra-cluster compactness (keep cluster members close)
  - Maximize inter-cluster separation (push cluster centers apart)
- **Benefit**: Creates more distinct and interpretable customer segments

### 3. Intelligent Initialization
- **Problem**: FDEKM is sensitive to random initialization
- **Solution**: Uses K-Means++ on pretrained embedding space for initial cluster centers
- **Benefit**: Faster and more stable convergence

## Architecture

### Components

1. **Attention Encoder (`AttentionEncoder`)**
   - Attention Layer: Learns feature importance weights α(x) = [α_L, α_R, α_F, α_M]
   - Feature Weighting: Multiplies input with attention weights (element-wise)
   - Deep Encoder: Transforms weighted features into embedding space

2. **Fuzzy Clustering Layer**
   - Computes fuzzy membership matrix based on distances to cluster centers
   - Uses standard FCM formulation with learnable cluster centers

3. **Enhanced Loss Function**
   - L_fuzzy: Standard FCM loss for intra-cluster compactness
   - L_sep: Cluster separation loss (sum of inverse squared distances between centers)
   - L_attention: Attention regularization to encourage diverse attention weights
   - Total: L_A-FDEKM = L_fuzzy + γ * L_sep + α * L_attention

## Algorithm Workflow

### Step 0: Intelligent Initialization
1. Pretrain standard autoencoder without attention layer
2. Encode entire dataset into initial embedding space
3. Run K-Means++ to find optimal initial cluster center locations
4. Initialize attention weights equally for all features

### Step 1: Concurrent Optimization Loop
For each iteration:

**Phase A: Update Fuzzy Clustering (Fixed Encoder)**
- Forward pass through attention encoder
- Update fuzzy membership matrix U based on current embeddings
- Update cluster centers V based on fuzzy memberships

**Phase B: Update Attention Encoder (Fixed Clustering)**
- Compute enhanced loss function value
- Backpropagate gradients through attention encoder
- Update encoder parameters using Adam optimizer

### Step 2: Output
- Customer segments: Final fuzzy membership matrix and cluster centers
- Trained encoder: Can classify new customers
- Feature importance: Learned attention weights for interpretability

## Usage

```python
from fuzzy_clustering.algorithms.afdekm import run_afdekm
import torch

# Prepare data (LRFM format)
X_tensor = torch.tensor(X, dtype=torch.float32)

# Run A-FDEKM
labels, embeddings, membership_matrix, cluster_centers, attention_weights, metrics = run_afdekm(
    X_tensor=X_tensor,
    X_for_metrics=X,
    k=3,                    # Number of clusters
    Iter=30,               # Number of optimization iterations
    hidden_dim_ae=10,      # Embedding dimension
    m=2.0,                 # Fuzzification parameter
    gamma=0.1,             # Weight for separation loss
    alpha=0.01,            # Weight for attention regularization
    lr=1e-3,               # Learning rate
    pretrain_epochs=25,    # Pretraining epochs
    verbose=True
)

# Analyze results
print("Cluster labels:", labels)
print("Average attention weights:", metrics['attention_weights_mean'])
print("Inter-cluster separation:", metrics['mean_inter_cluster_distance'])
```

## Key Parameters

- `k`: Number of clusters
- `m`: Fuzzification parameter (2.0 for standard fuzzy clustering)
- `gamma`: Weight for separation loss (0.1 recommended)
- `alpha`: Weight for attention regularization (0.01 recommended)
- `Iter`: Number of optimization iterations (30-50 recommended)
- `pretrain_epochs`: Autoencoder pretraining epochs (20-30 recommended)

## Output Metrics

A-FDEKM provides comprehensive evaluation metrics:

- **Standard clustering metrics**: Silhouette score, Calinski-Harabasz index, etc.
- **Fuzzy clustering metrics**: Partition coefficient, partition entropy
- **A-FDEKM specific metrics**:
  - `attention_weights_mean`: Average importance of each feature
  - `attention_weights_std`: Variability in attention weights
  - `mean_inter_cluster_distance`: Average separation between clusters
  - `min_inter_cluster_distance`: Minimum separation between clusters
  - `final_objective`: Final loss function value

## Interpretability

The attention mechanism provides interpretability through learned feature weights:

```python
# Get feature importance
feature_names = ['Latency', 'Recency', 'Frequency', 'Monetary']
avg_attention = metrics['attention_weights_mean']

for feature, weight in zip(feature_names, avg_attention):
    print(f"{feature}: {weight:.4f}")

# Find most important feature
most_important = feature_names[np.argmax(avg_attention)]
print(f"Most important feature: {most_important}")
```

## Integration with Main Pipeline

A-FDEKM is integrated into the main fuzzy clustering pipeline and can be run alongside other algorithms:

```python
from fuzzy_clustering.main import main

# Run all algorithms including A-FDEKM
main()
```

The algorithm will be executed on all supported datasets (USPS, E-commerce, Country) and results will be saved to `clustering_results.csv`.

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Scikit-learn >= 0.24.0
- Pandas >= 1.3.0

## Testing

Run the test script to see A-FDEKM in action:

```bash
python test_afdekm.py
```

This will generate synthetic LRFM data, compare A-FDEKM with FDEKM, and create visualization plots showing the improvements.
