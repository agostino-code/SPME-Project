"""Check actual FCM convergence iterations on each dataset."""
import numpy as np
import skfuzzy as fuzz
from dataset import Dataset

datasets = [
    ("../Software-estimation-datasets/albrecht.arff", "Effort", "Albrecht"),
    ("../Software-estimation-datasets/china.arff", "Effort", "China"),
    ("../Software-estimation-datasets/maxwell.arff", "Effort", "Maxwell"),
    ("../Software-estimation-datasets/Desharnais.csv", "Effort", "Desharnais"),
    ("../Software-estimation-datasets/COCOMO-81.csv", "actual", "COCOMO-81"),
]

for path, target, name in datasets:
    ds = Dataset(path, target_column_name=target)
    n_feat, n_samp = ds.training_data.shape
    print(f"\n{'='*50}")
    print(f"{name}: {n_samp} train samples, {n_feat} features")
    
    for n_clusters in [2, 3, 4, 5]:
        if n_clusters >= n_samp:
            continue
        # Run cmeans and track iterations
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            ds.training_data,
            c=n_clusters,
            m=2.0,
            error=0.005,
            maxiter=1000
        )
        n_iters = len(jm) - 1  # jm has initial + each iteration
        print(f"  k={n_clusters}: converged in {n_iters} iterations (maxiter=1000), fpc={fpc:.4f}")
