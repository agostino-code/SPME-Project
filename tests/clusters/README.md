# Test 1: Cluster Validation

## Description
This test evaluates fuzzy c-means (FCM) cluster validation metrics to replace/complement the Xie-Beni (XB) index:
- **Silhouette Score:** Higher is better (range [-1, 1]).
- **Davies-Bouldin Index (DB):** Lower is better.
- **Calinski-Harabasz Index (CH):** Higher is better.
- **End-to-End LOOCV Performance:** Validates if geometric cluster metrics align with minimal MedAE/MAE.

## Usage
Run the experiment:
```bash
python tests/clusters/run_clusters_test.py
```
Output files:
- `cluster_metrics.csv`: Metrics per dataset, cluster count $k$, and fuzzifier $m$.
- `recommendations.json`: Best $k$ recommendation per dataset.
