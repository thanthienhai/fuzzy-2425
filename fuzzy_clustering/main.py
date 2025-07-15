"""
Main execution module for fuzzy clustering algorithms.

This module provides the main entry point for running various clustering
algorithms on different datasets and collecting evaluation results.
"""

import os
import pandas as pd
from .data import load_usps_data, load_ecommerce_data, load_country_data
from .data import preprocess_usps_for_dekm, preprocess_usps_for_pfcm
from .algorithms import (
    run_dekm, run_pfcm, run_fcm, run_kmeans_standalone, 
    run_dbscan, run_cfcm, run_fdekm
)
from .metrics.fuzzy_metrics import pci_index, fhv_index, xbi_index


def load_custom_metrics(base_path='.'):
    """
    Load custom evaluation metrics from 'tools' directory.
    
    Args:
        base_path: Base directory path where tools folder is located
    """
    global pci_index, fhv_index, xbi_index
    tools_dir = os.path.join(base_path, 'tools')
    try:
        with open(os.path.join(tools_dir, 'pci.py'), 'r') as f_pci:
            exec(f_pci.read(), globals())
        with open(os.path.join(tools_dir, 'fhv.py'), 'r') as f_fhv:
            exec(f_fhv.read(), globals())
        with open(os.path.join(tools_dir, 'xbi.py'), 'r') as f_xbi:
            exec(f_xbi.read(), globals())
        print(f"Successfully loaded custom evaluation metrics from '{tools_dir}'.")
    except FileNotFoundError:
        print(f"Warning: One or more evaluation tool files (pci.py, fhv.py, xbi.py) not found in '{tools_dir}'.")
        print("Custom metrics (PCI, FHV, XBI) will default to NaN.")
    except Exception as e:
        print(f"Error loading custom metrics from '{tools_dir}': {e}. Custom metrics will default to NaN.")


def run_algorithms_on_usps(base_path='.', verbose=True):
    """
    Run all clustering algorithms on the USPS dataset.
    
    Args:
        base_path: Base directory path where data files are located
        verbose: Whether to print detailed progress information
        
    Returns:
        list: List of result dictionaries for each algorithm
    """
    results = []
    
    print("\nProcessing USPS Dataset...")
    X_usps_flat, y_usps = load_usps_data(base_path)
    
    if X_usps_flat is not None:
        usps_n_clusters = 10 
        usps_input_dim = X_usps_flat.shape[1]

        # DEKM on USPS
        print("\nRunning DEKM on USPS...")
        try:
            X_usps_tensor, X_usps_scaled_np = preprocess_usps_for_dekm(X_usps_flat.copy())
            if X_usps_tensor is not None:
                dekm_labels, _, dekm_metrics = run_dekm(
                    X_usps_tensor, X_usps_scaled_np, k=usps_n_clusters, Iter=3, 
                    input_dim=usps_input_dim, hidden_dim_ae=usps_n_clusters, verbose=verbose
                )
                results.append({'dataset': 'USPS', 'algorithm': 'DEKM', **dekm_metrics})
                print("DEKM on USPS results:", dekm_metrics)
            else:
                results.append({'dataset': 'USPS', 'algorithm': 'DEKM', 'error': 'Preprocessing failed'})
        except Exception as e:
            print(f"Error running DEKM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'DEKM', 'error': str(e)})

        # PFCM on USPS
        print("\nRunning PFCM on USPS...")
        try:
            X_usps_pca = preprocess_usps_for_pfcm(X_usps_flat.copy())
            if X_usps_pca is not None:
                _, _, _, _, pfcm_metrics = run_pfcm(
                    X_usps_pca, n_clusters=usps_n_clusters, verbose=verbose
                )
                results.append({'dataset': 'USPS', 'algorithm': 'PFCM', **pfcm_metrics})
                print("PFCM on USPS results:", pfcm_metrics)
            else:
                results.append({'dataset': 'USPS', 'algorithm': 'PFCM', 'error': 'Preprocessing failed'})
        except Exception as e:
            print(f"Error running PFCM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'PFCM', 'error': str(e)})

        # FCM on USPS
        print("\nRunning FCM on USPS...")
        try:
            _, X_usps_fcm_input = preprocess_usps_for_dekm(X_usps_flat.copy())
            if X_usps_fcm_input is not None:
                _, _, _, fcm_metrics_usps = run_fcm(
                    X_usps_fcm_input, n_clusters=usps_n_clusters, verbose=verbose
                )
                results.append({'dataset': 'USPS', 'algorithm': 'FCM', **fcm_metrics_usps})
                print("FCM on USPS results:", fcm_metrics_usps)
            else:
                results.append({'dataset': 'USPS', 'algorithm': 'FCM', 'error': 'Preprocessing failed'})
        except Exception as e:
            print(f"Error running FCM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'FCM', 'error': str(e)})

        # K-Means on USPS
        print("\nRunning K-Means (Standalone) on USPS...")
        try:
            if X_usps_scaled_np is not None:
                _, _, kmeans_metrics_usps = run_kmeans_standalone(
                    X_usps_scaled_np, n_clusters=usps_n_clusters, verbose=verbose
                )
                results.append({'dataset': 'USPS', 'algorithm': 'KMeans', **kmeans_metrics_usps})
                print("K-Means on USPS results:", kmeans_metrics_usps)
            else:
                results.append({'dataset': 'USPS', 'algorithm': 'KMeans', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running K-Means on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'KMeans', 'error': str(e)})

        # DBSCAN on USPS
        print("\nRunning DBSCAN on USPS...")
        try:
            if X_usps_scaled_np is not None:
                _, dbscan_metrics_usps = run_dbscan(
                    X_usps_scaled_np, eps=2.0, min_samples=10, verbose=verbose
                )
                results.append({'dataset': 'USPS', 'algorithm': 'DBSCAN', **dbscan_metrics_usps})
                print("DBSCAN on USPS results:", dbscan_metrics_usps)
            else:
                results.append({'dataset': 'USPS', 'algorithm': 'DBSCAN', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running DBSCAN on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'DBSCAN', 'error': str(e)})

        # CFCM on USPS (simplified version)
        print("\nRunning CFCM on USPS...")
        try:
            if X_usps_scaled_np is not None:
                _, _, _, cfcm_metrics_usps = run_cfcm(
                    X_usps_scaled_np, n_clusters=usps_n_clusters, verbose=verbose
                )
                results.append({'dataset': 'USPS', 'algorithm': 'CFCM', **cfcm_metrics_usps})
                print("CFCM on USPS results:", cfcm_metrics_usps)
            else:
                results.append({'dataset': 'USPS', 'algorithm': 'CFCM', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running CFCM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'CFCM', 'error': str(e)})

        # FDEKM on USPS (simplified version)
        print("\nRunning FDEKM on USPS...")
        try:
            if X_usps_tensor is not None:
                _, _, _, _, fdekm_metrics_usps = run_fdekm(
                    X_usps_tensor, X_usps_scaled_np, k=usps_n_clusters, Iter=3,
                    input_dim=usps_input_dim, hidden_dim_ae=usps_n_clusters, verbose=verbose
                )
                results.append({'dataset': 'USPS', 'algorithm': 'FDEKM', **fdekm_metrics_usps})
                print("FDEKM on USPS results:", fdekm_metrics_usps)
            else:
                results.append({'dataset': 'USPS', 'algorithm': 'FDEKM', 'error': 'Preprocessing failed'})
        except Exception as e:
            print(f"Error running FDEKM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'FDEKM', 'error': str(e)})

    else:
        print("USPS dataset could not be loaded. Skipping USPS experiments.")
    
    return results


def run_algorithms_on_ecommerce(base_path='.', verbose=True):
    """
    Run clustering algorithms on the e-commerce dataset.

    Args:
        base_path: Base directory path where data files are located
        verbose: Whether to print detailed progress information

    Returns:
        list: List of result dictionaries for each algorithm
    """
    results = []

    print("\nProcessing E-commerce Dataset...")
    X_ecommerce = load_ecommerce_data(base_path)

    if X_ecommerce is not None:
        ecommerce_n_clusters = 3  # Typical for e-commerce customer segmentation

        # Run a subset of algorithms on e-commerce data
        algorithms_to_run = [
            ('FCM', lambda: run_fcm(X_ecommerce, n_clusters=ecommerce_n_clusters, verbose=verbose)),
            ('KMeans', lambda: run_kmeans_standalone(X_ecommerce, n_clusters=ecommerce_n_clusters, verbose=verbose)),
            ('DBSCAN', lambda: run_dbscan(X_ecommerce, eps=0.5, min_samples=5, verbose=verbose))
        ]

        for algo_name, algo_func in algorithms_to_run:
            print(f"\nRunning {algo_name} on E-commerce...")
            try:
                result = algo_func()
                if len(result) >= 3:  # Has metrics
                    metrics = result[-1] if isinstance(result[-1], dict) else {}
                    results.append({'dataset': 'E-commerce', 'algorithm': algo_name, **metrics})
                    print(f"{algo_name} on E-commerce results:", metrics)
                else:
                    results.append({'dataset': 'E-commerce', 'algorithm': algo_name, 'error': 'Invalid result format'})
            except Exception as e:
                print(f"Error running {algo_name} on E-commerce: {e}")
                results.append({'dataset': 'E-commerce', 'algorithm': algo_name, 'error': str(e)})
    else:
        print("E-commerce dataset could not be loaded. Skipping e-commerce experiments.")

    return results


def run_algorithms_on_country(base_path='.', verbose=True):
    """
    Run clustering algorithms on the country development dataset.

    Args:
        base_path: Base directory path where data files are located
        verbose: Whether to print detailed progress information

    Returns:
        list: List of result dictionaries for each algorithm
    """
    results = []

    print("\nProcessing Country Dataset...")
    country_data = load_country_data(base_path)

    if country_data[0] is not None:
        X_country, country_names = country_data
        country_n_clusters = 4  # Typical for country development clustering

        # Run a subset of algorithms on country data
        algorithms_to_run = [
            ('FCM', lambda: run_fcm(X_country, n_clusters=country_n_clusters, verbose=verbose)),
            ('KMeans', lambda: run_kmeans_standalone(X_country, n_clusters=country_n_clusters, verbose=verbose)),
            ('DBSCAN', lambda: run_dbscan(X_country, eps=1.0, min_samples=3, verbose=verbose))
        ]

        for algo_name, algo_func in algorithms_to_run:
            print(f"\nRunning {algo_name} on Country...")
            try:
                result = algo_func()
                if len(result) >= 3:  # Has metrics
                    metrics = result[-1] if isinstance(result[-1], dict) else {}
                    results.append({'dataset': 'Country', 'algorithm': algo_name, **metrics})
                    print(f"{algo_name} on Country results:", metrics)
                else:
                    results.append({'dataset': 'Country', 'algorithm': algo_name, 'error': 'Invalid result format'})
            except Exception as e:
                print(f"Error running {algo_name} on Country: {e}")
                results.append({'dataset': 'Country', 'algorithm': algo_name, 'error': str(e)})
    else:
        print("Country dataset could not be loaded. Skipping country experiments.")

    return results


def save_results(results, output_file='clustering_results.csv'):
    """
    Save clustering results to a CSV file.

    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Total experiments: {len(results)}")

        # Print summary
        print("\nSummary by dataset and algorithm:")
        if 'dataset' in df.columns and 'algorithm' in df.columns:
            summary = df.groupby(['dataset', 'algorithm']).size().reset_index(name='count')
            print(summary.to_string(index=False))
    else:
        print("No results to save.")


def main():
    """
    Main function to run all clustering algorithms on all datasets.
    """
    results = []
    verbose_run = True
    base_path = '.'  # Assuming script is run from project root

    # Load custom evaluation metrics from 'tools' directory
    load_custom_metrics(base_path)

    # Run algorithms on all datasets
    results.extend(run_algorithms_on_usps(base_path, verbose_run))
    results.extend(run_algorithms_on_ecommerce(base_path, verbose_run))
    results.extend(run_algorithms_on_country(base_path, verbose_run))

    # Save results
    save_results(results)

    print("\n" + "="*50)
    print("All clustering experiments completed!")
    print("="*50)


if __name__ == '__main__':
    main()
