import os
import sys
import json
import itertools
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fuzzy-DCNN-implementation')))

from utils import DATASET_PATHS, run_loocv_experiment

def run_2stage_gridsearch():
    print("=" * 80)
    print("TEST 3: EFFICIENT 2-STAGE HYPERPARAMETER SEARCH")
    print("=" * 80)

    output_dir = os.path.dirname(__file__)
    best_config_file = os.path.join(output_dir, 'best_hyperparameters.json')

    # Stage 1 search grid (Architecture & Regularization)
    stage1_grid = {
        'num_filters': [8, 16, 32],
        'dropout_rate': [0.1, 0.2, 0.3],
        'weight_decay': [0.0, 0.001],
        'cluster_method': [3, 8, 'XB-index', 'silhouette']
    }

    # Stage 2 search grid (Optimization & Training)
    stage2_grid = {
        'dcnn_lr': [0.005, 0.001, 0.0005, 0.0001],
        'dcnn_batch_size': [8, 16, 32],
        'dcnn_epochs': [200, 500, 1000]
    }

    all_best_configs = {}

    for ds_name, ds_path in DATASET_PATHS.items():
        print(f"\n==================================================")
        print(f"GRID SEARCH FOR DATASET: {ds_name.upper()}")
        print(f"==================================================")

        results_csv = os.path.join(output_dir, f'grid_results_{ds_name}.csv')
        dataset_records = []

        # Default baseline stage 2 params during stage 1
        curr_lr = 0.001
        curr_batch = 16
        curr_epochs = 300

        best_s1_medae = float('inf')
        best_s1_config = {'num_filters': 16, 'dropout_rate': 0.2, 'weight_decay': 0.0, 'cluster_method': 3}

        print("\n--- STAGE 1: Architecture & Regularization Search ---")
        s1_keys, s1_vals = zip(*stage1_grid.items())
        for combo in itertools.product(*s1_vals):
            params = dict(zip(s1_keys, combo))
            m, _, _ = run_loocv_experiment(
                dataset_path=ds_path,
                seed=42,
                dcnn_lr=curr_lr,
                dcnn_batch_size=curr_batch,
                dcnn_epochs=curr_epochs,
                num_filters=params['num_filters'],
                dropout_rate=params['dropout_rate'],
                weight_decay=params['weight_decay'],
                cluster_method=params['cluster_method']
            )

            record = {'stage': 1, **params, 'dcnn_lr': curr_lr, 'dcnn_batch_size': curr_batch, 'dcnn_epochs': curr_epochs, **m}
            dataset_records.append(record)

            print(f"  Filters: {params['num_filters']:2d} | Dropout: {params['dropout_rate']:.1f} | L2: {params['weight_decay']:.4f} | Cluster Method: {params['cluster_method']} => MedAE: {m['medae']:8.2f} | MAE: {m['mae']:8.2f} | SA: {m['sa']:.4f}")

            if m['medae'] < best_s1_medae:
                best_s1_medae = m['medae']
                best_s1_config = params

        print(f"\n>>> Best Stage 1 Config for {ds_name}: {best_s1_config} (MedAE: {best_s1_medae:.2f})")

        print("\n--- STAGE 2: Optimization & Training Search ---")
        best_s2_medae = float('inf')
        best_full_config = {**best_s1_config, 'dcnn_lr': curr_lr, 'dcnn_batch_size': curr_batch, 'dcnn_epochs': curr_epochs}

        s2_keys, s2_vals = zip(*stage2_grid.items())
        for combo in itertools.product(*s2_vals):
            params = dict(zip(s2_keys, combo))
            m, _, _ = run_loocv_experiment(
                dataset_path=ds_path,
                seed=42,
                num_filters=best_s1_config['num_filters'],
                dropout_rate=best_s1_config['dropout_rate'],
                weight_decay=best_s1_config['weight_decay'],
                cluster_method=best_s1_config['cluster_method'],
                dcnn_lr=params['dcnn_lr'],
                dcnn_batch_size=params['dcnn_batch_size'],
                dcnn_epochs=params['dcnn_epochs']
            )

            record = {'stage': 2, **best_s1_config, **params, **m}
            dataset_records.append(record)

            print(f"  LR: {params['dcnn_lr']:.4f} | Batch: {params['dcnn_batch_size']:2d} | Epochs: {params['dcnn_epochs']:4d} => MedAE: {m['medae']:8.2f} | MAE: {m['mae']:8.2f} | SA: {m['sa']:.4f}")

            if m['medae'] < best_s2_medae:
                best_s2_medae = m['medae']
                best_full_config = {**best_s1_config, **params}

        df_ds = pd.DataFrame(dataset_records)
        df_ds.to_csv(results_csv, index=False)
        print(f"\nSaved dataset grid search results to {results_csv}")

        all_best_configs[ds_name] = {
            'best_medae': best_s2_medae,
            'hyperparameters': best_full_config
        }

    with open(best_config_file, 'w') as f:
        json.dump(all_best_configs, f, indent=2)
    print(f"\nSaved global best hyperparameter configurations to {best_config_file}")

if __name__ == '__main__':
    run_2stage_gridsearch()
