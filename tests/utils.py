import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import LeaveOneOut, KFold
from joblib import Parallel, delayed

# Add fuzzy-DCNN-implementation to python path
IMPLEMENTATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fuzzy-DCNN-implementation'))
if IMPLEMENTATION_DIR not in sys.path:
    sys.path.append(IMPLEMENTATION_DIR)

from dataset import Dataset
from fold_worker import is_china_dataset, target_column_for, process_single_fold

DATASET_PATHS = {
    'cocomo': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Software-estimation-datasets', 'COCOMO-81.csv')),
    'albrecht': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Software-estimation-datasets', 'albrecht.arff')),
    'desharnais': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Software-estimation-datasets', 'Desharnais.csv')),
    'maxwell': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Software-estimation-datasets', 'maxwell.arff')),
    'china': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Software-estimation-datasets', 'china.arff')),
}

def resolve_dataset_path(dataset_key_or_path: str) -> str:
    key_lower = os.path.basename(dataset_key_or_path).lower().split('.')[0]
    if key_lower in DATASET_PATHS:
        return DATASET_PATHS[key_lower]
    if os.path.exists(dataset_key_or_path):
        return os.path.abspath(dataset_key_or_path)
    raise FileNotFoundError(f"Dataset not found: {dataset_key_or_path}")

def set_global_seed(seed: int = 2501439):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(actual, predicted, n_random_baseline_runs=1000):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    abs_errors = np.abs(actual - predicted)
    mae = float(np.mean(abs_errors))
    medae = float(np.median(abs_errors))
    dead_neuron_rate = float(np.mean(predicted <= 0.0) * 100.0)

    # Random baseline estimation for SA & Delta
    random_maes = np.zeros(n_random_baseline_runs)
    for i in range(n_random_baseline_runs):
        np.random.seed(i)
        pred_random = np.random.permutation(actual)
        random_maes[i] = np.mean(np.abs(actual - pred_random))

    mae_random = np.mean(random_maes)
    sd_random = np.std(random_maes)

    sa = float(1.0 - (mae / mae_random)) if mae_random != 0 else 0.0
    delta = float((mae_random - mae) / sd_random) if sd_random != 0 else 0.0

    return {
        'mae': mae,
        'medae': medae,
        'sa': sa,
        'delta': delta,
        'dead_neuron_rate': dead_neuron_rate,
        'abs_errors': abs_errors
    }

def run_loocv_experiment(dataset_path: str,
                         seed: int = 2501439,
                         use_knn_imputation: bool = False,
                         max_fuzzy_clusters: int = 10,
                         dcnn_lr: float = 0.001,
                         dcnn_epochs: int = 1000,
                         dcnn_batch_size: int = 16,
                         num_filters: int = 16,
                         dropout_rate: float = 0.2,
                         weight_decay: float = 0.0,
                         loss_type: str = 'huber',
                         huber_delta: float = 0.01,
                         activation_type: str = 'linear_clip',
                         fuzzy_m: float = 2.0,
                         use_log_target: bool = False,
                         cluster_method: str | int = 'XB-index',
                         n_jobs: int = None):
    
    full_path = resolve_dataset_path(dataset_path)
    target_col = target_column_for(full_path)
    ds_full = Dataset(full_path, target_column_name=target_col, use_knn_imputation=use_knn_imputation, verbose=False)

    if is_china_dataset(full_path):
        cv = KFold(n_splits=10, shuffle=True, random_state=2501439)
    else:
        cv = LeaveOneOut()

    use_gpu = is_china_dataset(full_path) and torch.cuda.is_available()
    num_jobs = n_jobs if n_jobs is not None else (1 if use_gpu else -1)

    fold_results = Parallel(n_jobs=num_jobs, backend='loky', verbose=0)(
        delayed(process_single_fold)(
            dataset_path=full_path,
            target_col=target_col,
            use_knn_imputation=use_knn_imputation,
            train_idx=train_idx,
            test_idx=test_idx,
            random_seed=seed,
            max_fuzzy_clusters=max_fuzzy_clusters,
            dcnn_lr=dcnn_lr,
            dcnn_epochs=dcnn_epochs,
            dcnn_batch_size=dcnn_batch_size,
            num_filters=num_filters,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            loss_type=loss_type,
            huber_delta=huber_delta,
            activation_type=activation_type,
            fuzzy_m=fuzzy_m,
            use_log_target=use_log_target,
            cluster_method=cluster_method
        )
        for train_idx, test_idx in cv.split(ds_full.data)
    )

    actuals, predictions = [], []
    for fold_actuals, fold_predictions in fold_results:
        actuals.extend(fold_actuals)
        predictions.extend(fold_predictions)

    actuals = np.array(actuals, dtype=float)
    predictions = np.array(predictions, dtype=float)
    metrics = compute_metrics(actuals, predictions)
    
    return metrics, actuals, predictions

def compute_wilcoxon_test(actuals, preds_base, preds_variant):
    err_base = np.abs(actuals - preds_base)
    err_variant = np.abs(actuals - preds_variant)

    diff = err_base - err_variant
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))

    if np.all(err_base == err_variant):
        p_val = 1.0
    else:
        try:
            stat, p_val = stats.wilcoxon(err_base, err_variant)
        except Exception:
            p_val = 1.0

    return {
        'p_value': float(p_val),
        'statistically_significant': bool(p_val < 0.05),
        'wins': wins,
        'losses': losses,
        'ties': ties
    }
