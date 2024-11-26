import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functools import wraps

def plot_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        distances = func(*args, **kwargs)
        plt.plot(distances)
        plt.title('k-NN Distance Plot')
        plt.xlabel('Data Points')
        plt.ylabel('k-NN Distance')
        plt.show()
        return distances
    return wrapper

@plot_decorator
def analyze_knn(X, k=5):
    """
    Master function that combines compute_knn_distances and plot_knn_distances.
    Computes k-NN distances and automatically plots them.
    
    Args:
        X: Input data
        k: Number of neighbors (default=5)
    Returns:
        distances: Sorted k-NN distances
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    return np.sort(distances[:, k-1])

def optimize_dbscan_params(X, k=5, percentiles=[0.80, 0.85, 0.90, 0.95], 
                         min_samples_values=[5, 10, 15, 20]):
    """
    Master function that combines find_best_epsilon and test_min_samples.
    Performs a complete parameter optimization for DBSCAN.
    
    Args:
        X: Input data
        k: Number of neighbors for k-NN (default=5)
        percentiles: List of percentiles to try for epsilon
        min_samples_values: List of min_samples values to try
    Returns:
        tuple: (best_params, best_score, all_results)
            - best_params: (epsilon, min_samples) that gave best score
            - best_score: Best silhouette score achieved
            - all_results: Dictionary with all tested combinations and their scores
    """
    # First get the distances
    distances = analyze_knn(X, k)
    
    all_results = {}
    best_score = -1
    best_params = None
    
    # Try all combinations of epsilon (from percentiles) and min_samples
    for percentile in percentiles:
        epsilon = distances[round(len(distances) * percentile)]
        
        for min_samples in min_samples_values:
            db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
            labels = db.labels_
            
            if len(set(labels)) > 1:  # Only evaluate if we have more than one cluster
                score = silhouette_score(X, labels)
                all_results[(epsilon, min_samples)] = score
                
                if score > best_score:
                    best_score = score
                    best_params = (epsilon, min_samples)
    
    return best_params, best_score, all_results

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    X = np.random.rand(4312, 3)
    
    # Get and plot k-NN distances
    distances = analyze_knn(X)
    
    # Find optimal parameters
    best_params, best_score, all_results = optimize_dbscan_params(X)
    print(f"Best parameters (epsilon, min_samples): {best_params}")
    print(f"Best silhouette score: {best_score}")
