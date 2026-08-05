import numpy as np
from fuzzy import FuzzyCMeans

def fixed_fcm(dataset, k, fuzzifier_m=2.0):
    fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=k, m=fuzzifier_m)
    mu = fuzzy_cmeans.fit_transform()
    return fuzzy_cmeans, k, mu

def elbow_XB(dataset, max_clusters, fuzzifier_m=2.0):
        
    n_samples = dataset.training_data.shape[1]
    max_clusters = min(max_clusters, max(2, int(np.sqrt(n_samples))))
    
    current_best_XB = float('inf')
    current_best_fuzzy_cmeans = None
    current_best_mu = None
    current_best_cluster_n = 0
    
    for n_clusters in range(2, max_clusters + 1):
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=n_clusters, m=fuzzifier_m)
        mu = fuzzy_cmeans.fit_transform()
        
        current_XB = fuzzy_cmeans.XB()
        if current_XB < current_best_XB:
            current_best_XB = current_XB
            current_best_fuzzy_cmeans = fuzzy_cmeans
            current_best_mu = mu
            current_best_cluster_n = n_clusters

    assert current_best_fuzzy_cmeans is not None and current_best_mu is not None and current_best_cluster_n > 0, "No valid fuzzy c-means model found."

    return current_best_fuzzy_cmeans, current_best_cluster_n, current_best_mu

def elbow_silhouette(dataset, max_clusters, fuzzifier_m=2.0):
    n_samples = dataset.training_data.shape[1]
    max_clusters = min(max_clusters, max(2, int(np.sqrt(n_samples))))
    
    current_best_silhouette = float('-inf')
    current_best_fuzzy_cmeans = None
    current_best_mu = None
    current_best_cluster_n = 0
    
    for n_clusters in range(2, max_clusters + 1):
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=n_clusters, m=fuzzifier_m)
        mu = fuzzy_cmeans.fit_transform()
        
        current_silhouette = fuzzy_cmeans.silhouette_score()
        if current_silhouette > current_best_silhouette:
            current_best_silhouette = current_silhouette
            current_best_fuzzy_cmeans = fuzzy_cmeans
            current_best_mu = mu
            current_best_cluster_n = n_clusters

    assert current_best_fuzzy_cmeans is not None and current_best_mu is not None and current_best_cluster_n > 0, "No valid fuzzy c-means model found."

    return current_best_fuzzy_cmeans, current_best_cluster_n, current_best_mu