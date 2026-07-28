import numpy as np
from fuzzy import FuzzyCMeans

def elbow_XB(dataset, max_clusters):
    n_samples = dataset.training_data.shape[1]
    max_clusters = min(max_clusters, max(2, int(np.sqrt(n_samples))))
    
    current_best_XB = float('inf')
    current_best_fuzzy_cmeans = None
    current_best_mu = None
    current_best_cluster_n = 0
    
    for n_clusters in range(2, max_clusters + 1):
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=n_clusters)
        mu = fuzzy_cmeans.fit_transform()
        
        current_XB = fuzzy_cmeans.XB()
        if current_XB < current_best_XB:
            current_best_XB = current_XB
            current_best_fuzzy_cmeans = fuzzy_cmeans
            current_best_mu = mu
            current_best_cluster_n = n_clusters

    assert current_best_fuzzy_cmeans is not None and current_best_mu is not None and current_best_cluster_n > 0, "No valid fuzzy c-means model found."

    return current_best_fuzzy_cmeans, current_best_cluster_n, current_best_mu
