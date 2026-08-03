import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fuzzy-DCNN-implementation')))

from utils import DATASET_PATHS, resolve_dataset_path, run_loocv_experiment
from dataset import Dataset
from fuzzy import FuzzyCMeans

def run_cluster_validation():
    print("=" * 80)
    print("TEST 1: CLUSTER VALIDATION & FUZZIFIER TUNING")
    print("=" * 80)

    output_dir = os.path.dirname(__file__)
    results_csv = os.path.join(output_dir, 'cluster_metrics.csv')
    rec_json = os.path.join(output_dir, 'recommendations.json')

    metrics_records = []
    recommendations = {}

    for ds_name, ds_path in DATASET_PATHS.items():
        print(f"\n---> Validating clusters for dataset: {ds_name.upper()}")
        full_path = resolve_dataset_path(ds_path)
        target_col = 'actual' if 'cocomo' in ds_name else 'Effort'

        ds_full = Dataset(full_path, target_column_name=target_col, verbose=False)
        X_data = ds_full.training_data.T  # shape (n_samples, n_features)
        n_samples = X_data.shape[0]

        best_sil = -1.0
        best_k_sil = 2
        best_db = float('inf')
        best_k_db = 2
        best_loocv_medae = float('inf')
        best_k_loocv = 2

        max_k = min(10, max(3, int(np.sqrt(n_samples))))

        for m in [1.5, 2.0, 2.5]:
            for k in range(2, max_k + 1):
                fuzzy = FuzzyCMeans(ds_full, num_clusters=k, m=m)
                fuzzy.fit()
                
                memberships = fuzzy.membership_degrees
                hard_labels = np.argmax(memberships, axis=1)

                if len(np.unique(hard_labels)) > 1:
                    sil = float(silhouette_score(X_data, hard_labels))
                    db = float(davies_bouldin_score(X_data, hard_labels))
                    ch = float(calinski_harabasz_score(X_data, hard_labels))
                else:
                    sil, db, ch = -1.0, float('inf'), 0.0

                xb_idx = float(fuzzy.XB())

                # End-to-end LOOCV test for selected m=2.0
                if m == 2.0 and k in [2, 3, 4, 5, 6]:
                    loocv_m, _, _ = run_loocv_experiment(
                        dataset_path=full_path,
                        seed=42,
                        dcnn_epochs=300,
                        force_k=k,
                        fuzzy_m=m
                    )
                    mae_loocv = loocv_m['mae']
                    medae_loocv = loocv_m['medae']
                    sa_loocv = loocv_m['sa']
                else:
                    mae_loocv, medae_loocv, sa_loocv = None, None, None

                record = {
                    'dataset': ds_name,
                    'k': k,
                    'm': m,
                    'silhouette_score': sil,
                    'davies_bouldin_index': db,
                    'calinski_harabasz_index': ch,
                    'xie_beni_index': xb_idx,
                    'loocv_mae': mae_loocv,
                    'loocv_medae': medae_loocv,
                    'loocv_sa': sa_loocv
                }
                metrics_records.append(record)
                print(f"  k={k}, m={m:.1f} | Sil: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f} | XB: {xb_idx:.4f} | LOOCV MedAE: {medae_loocv}")

                if m == 2.0:
                    if sil > best_sil:
                        best_sil, best_k_sil = sil, k
                    if db < best_db:
                        best_db, best_k_db = db, k
                    if medae_loocv is not None and medae_loocv < best_loocv_medae:
                        best_loocv_medae, best_k_loocv = medae_loocv, k

        recommendations[ds_name] = {
            'optimal_k_silhouette': best_k_sil,
            'optimal_k_davies_bouldin': best_k_db,
            'optimal_k_loocv_performance': best_k_loocv,
            'recommended_k': best_k_loocv
        }

    df_results = pd.DataFrame(metrics_records)
    df_results.to_csv(results_csv, index=False)
    print(f"\nSaved cluster metrics to {results_csv}")

    with open(rec_json, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"Saved recommendations to {rec_json}")

if __name__ == '__main__':
    run_cluster_validation()
