import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut, KFold
from joblib import Parallel, delayed

from dataset import Dataset
from fold_worker import is_china_dataset, target_column_for, process_single_fold

sys.stdout.reconfigure(line_buffering=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
CV_SPLIT_SEED = 42
N_RANDOM_BASELINE_RUNS = 1000   # numero di permutazioni casuali usate per stimare SA / Delta
MAX_FUZZY_CLUSTERS = 10         # limite superiore per la ricerca del k ottimale

CHINA_KFOLD_SPLITS = 10         # China usa 10-fold invece di LOOCV

# Optimal hyperparameter configurations per dataset (empirically derived from SOTA test suite)
DATASET_BEST_PARAMS = {
    'cocomo-81.csv': {
        'cluster_method': 2,
        'num_filters': 8,
        'dropout_rate': 0.3,
        'weight_decay': 0.0,
        'dcnn_lr': 0.0005,
        'dcnn_batch_size': 16,
        'dcnn_epochs': 1000,
        'loss_type': 'huber',
        'huber_delta': 0.01,
        'activation_type': 'linear_clip',
        'use_log_target': True,
    },
    'albrecht.arff': {
        'cluster_method': 4,
        'num_filters': 32,
        'dropout_rate': 0.3,
        'weight_decay': 0.0,
        'dcnn_lr': 0.005,
        'dcnn_batch_size': 32,
        'dcnn_epochs': 200,
        'loss_type': 'rmse',
        'huber_delta': 0.5,
        'activation_type': 'linear_clip',
        'use_log_target': False,
    },
    'desharnais.csv': {
        'cluster_method': 3,
        'num_filters': 8,
        'dropout_rate': 0.2,
        'weight_decay': 0.0,
        'dcnn_lr': 0.0005,
        'dcnn_batch_size': 32,
        'dcnn_epochs': 200,
        'loss_type': 'huber',
        'huber_delta': 0.01,
        'activation_type': 'linear_clip',
        'use_log_target': True,
    },
    'maxwell.arff': {
        'cluster_method': 2,
        'num_filters': 32,
        'dropout_rate': 0.1,
        'weight_decay': 0.0,
        'dcnn_lr': 0.001,
        'dcnn_batch_size': 8,
        'dcnn_epochs': 200,
        'loss_type': 'huber',
        'huber_delta': 0.1,
        'activation_type': 'linear_clip',
        'use_log_target': False,
    },
    'china.arff': {
        'cluster_method': 5,
        'num_filters': 32,
        'dropout_rate': 0.1,
        'weight_decay': 0.0,
        'dcnn_lr': 0.001,
        'dcnn_batch_size': 16,
        'dcnn_epochs': 1000,
        'loss_type': 'huber',
        'huber_delta': 0.01,
        'activation_type': 'linear_clip',
        'use_log_target': True,
    }
}

DATASETS_TO_EVALUATE = [
    '../Software-estimation-datasets/COCOMO-81.csv',
    '../Software-estimation-datasets/albrecht.arff',
    '../Software-estimation-datasets/Desharnais.csv',
    '../Software-estimation-datasets/maxwell.arff',
    '../Software-estimation-datasets/china.arff',
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(actual, predicted):
    """
    Calcola MAE, MedAE, Standardized Accuracy (SA) ed effect size (Delta), confrontando
    l'errore del modello con quello di N_RANDOM_BASELINE_RUNS predittori casuali.
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    abs_errors = np.abs(actual - predicted)
    mae = float(np.mean(abs_errors))
    medae = float(np.median(abs_errors))

    random_maes = np.zeros(N_RANDOM_BASELINE_RUNS)
    for i in range(N_RANDOM_BASELINE_RUNS):
        np.random.seed(i)
        pred_random = np.random.permutation(actual)
        random_maes[i] = np.mean(np.abs(actual - pred_random))

    mae_random = np.mean(random_maes)
    sd_random = np.std(random_maes)

    sa = 1.0 - (mae / mae_random) if mae_random != 0 else 0.0
    delta = (mae_random - mae) / sd_random if sd_random != 0 else 0.0

    return mae, medae, sa, delta


# --------------------------------------------------------------------------- #
# Full LOOCV/K-fold run for one dataset using dataset-specific optimal params
# --------------------------------------------------------------------------- #

def run_loocv(dataset_path: str, use_knn_imputation: bool = False,
              seed: int = None, n_jobs: int = None, params_override: dict = None):
    """
    Esegue la validazione completa (LOOCV, o 10-fold per China) su un dataset
    applicando i parametri ottimali specifici del dataset.
    """
    ds_name_key = os.path.basename(dataset_path).lower()
    ds_params = DATASET_BEST_PARAMS.get(ds_name_key, {}).copy()
    if params_override:
        ds_params.update(params_override)

    model_seed = seed if seed is not None else RANDOM_SEED
    loss_type = ds_params.get('loss_type', 'huber')
    cluster_method = ds_params.get('cluster_method', None)
    dcnn_lr = ds_params.get('dcnn_lr', 0.001)
    dcnn_epochs = ds_params.get('dcnn_epochs', 1000)
    dcnn_batch_size = ds_params.get('dcnn_batch_size', 16)
    num_filters = ds_params.get('num_filters', 16)
    dropout_rate = ds_params.get('dropout_rate', 0.2)
    weight_decay = ds_params.get('weight_decay', 0.0)
    huber_delta = ds_params.get('huber_delta', 0.01)
    activation_type = ds_params.get('activation_type', 'linear_clip')
    use_log_target = ds_params.get('use_log_target', False)

    print(f"\n--- Running LOOCV for {os.path.basename(dataset_path)} "
          f"(KNN={use_knn_imputation}, LogTarget={use_log_target}, Loss={loss_type}, k={cluster_method}) ---", flush=True)

    target_col = target_column_for(dataset_path)
    ds_full = Dataset(dataset_path, target_column_name=target_col,
                      use_knn_imputation=use_knn_imputation,
                      use_log_target=use_log_target, verbose=True)

    if is_china_dataset(dataset_path):
        cv_seed = CV_SPLIT_SEED if 'CV_SPLIT_SEED' in globals() else 42
        cv = KFold(n_splits=CHINA_KFOLD_SPLITS, shuffle=True, random_state=cv_seed)
        print("NOTE: Using 10-fold CV (not LOOCV) for China.", flush=True)
    else:
        cv = LeaveOneOut()

    use_gpu = is_china_dataset(dataset_path) and torch.cuda.is_available()
    num_jobs = n_jobs if n_jobs is not None else (1 if use_gpu else -1)

    fold_results = Parallel(n_jobs=num_jobs, backend='loky', verbose=5)(
        delayed(process_single_fold)(
            dataset_path=dataset_path,
            target_col=target_col,
            use_knn_imputation=use_knn_imputation,
            train_idx=train_idx,
            test_idx=test_idx,
            random_seed=model_seed,
            max_fuzzy_clusters=MAX_FUZZY_CLUSTERS,
            dcnn_lr=dcnn_lr,
            dcnn_epochs=dcnn_epochs,
            dcnn_batch_size=dcnn_batch_size,
            num_filters=num_filters,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            loss_type=loss_type,
            huber_delta=huber_delta,
            activation_type=activation_type,
            use_log_target=use_log_target,
            cluster_method=cluster_method,
        )
        for train_idx, test_idx in cv.split(ds_full.data)
    )

    actuals, predictions, fold_indices = [], [], []
    for fold_idx, (fold_actuals, fold_predictions) in enumerate(fold_results):
        actuals.extend(fold_actuals)
        predictions.extend(fold_predictions)
        fold_indices.extend([fold_idx] * len(fold_actuals))

    mae, medae, sa, delta = compute_metrics(actuals, predictions)

    df_results = pd.DataFrame({
        'fold_index': fold_indices,
        'actual': actuals,
        'predicted': predictions,
    })
    df_results['abs_error'] = np.abs(df_results['actual'] - df_results['predicted'])
    df_results['squared_error'] = df_results['abs_error'] ** 2

    ds_name_no_ext = os.path.splitext(os.path.basename(dataset_path))[0]
    config_str = f"FDCNN_{loss_type}_k{cluster_method}_seed{model_seed}"
    csv_filename = f"results_per_fold_{ds_name_no_ext}__{config_str}.csv"
    
    df_results.to_csv(csv_filename, index=False)
    print(f"Saved per-fold results to {csv_filename}")

    print(f"Results for {os.path.basename(dataset_path)}:")
    print(f"MAE:   {mae:.4f}")
    print(f"MedAE: {medae:.4f}")
    print(f"SA:    {sa:.4f}")
    print(f"Delta: {delta:.4f}")

    return mae, medae, sa, delta, csv_filename


# --------------------------------------------------------------------------- #
# Entry point: run all configured datasets and print summary table
# --------------------------------------------------------------------------- #

def main():
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print("Note: Using GPU ONLY for China.arff. CPU is faster for small datasets due to transfer overhead.")
    else:
        print("No GPU detected.")

    results = {}
    for ds_path in DATASETS_TO_EVALUATE:
        if not os.path.exists(ds_path):
            print(f"File not found: {ds_path}")
            continue

        start_time = time.time()
        mae, medae, sa, delta, _ = run_loocv(ds_path)
        elapsed = time.time() - start_time

        method = '10-fold' if is_china_dataset(ds_path) else 'LOOCV'
        results[os.path.basename(ds_path)] = {
            'MAE': mae, 'MedAE': medae, 'SA': sa, 'Delta': delta, 'Time': elapsed, 'Method': method,
        }
        print(f"Time for {os.path.basename(ds_path)}: {elapsed:.2f} seconds")

    print_summary_table(results)


def print_summary_table(results: dict):
    print("\n" + "=" * 90)
    print("FINAL SUMMARY (POST-OPTIMIZATION LOOCV / 10-FOLD)")
    print("=" * 90)
    print(f"{'Dataset':<15} | {'Method':<10} | {'MAE':<10} | {'MedAE':<10} | {'SA':<10} | "
          f"{'Delta':<10} | {'Time (s)':<10}")
    for name, r in results.items():
        print(f"{name:<15} | {r['Method']:<10} | {r['MAE']:<10.2f} | {r['MedAE']:<10.2f} | "
              f"{r['SA']:<10.2f} | {r['Delta']:<10.2f} | {r['Time']:<10.2f}")
    print("=" * 90)


if __name__ == '__main__':
    main()