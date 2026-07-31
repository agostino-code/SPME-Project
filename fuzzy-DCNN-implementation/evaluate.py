import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from scipy.stats import wilcoxon
import multiprocessing
from functools import partial

from dataset import Dataset
from elbow import elbow_XB
from DCNN import EffortEstimator

sys.stdout.reconfigure(line_buffering=True)

def compute_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = np.mean(np.abs(actual - predicted))
    
    n_runs = 1000
    random_maes = np.zeros(n_runs)
    for i in range(n_runs):
        np.random.seed(i)
        pred_random = np.random.permutation(actual)
        random_maes[i] = np.mean(np.abs(actual - pred_random))
        
    mae_random = np.mean(random_maes)
    sd_random = np.std(random_maes)
    
    sa = 1.0 - (mae / mae_random) if mae_random != 0 else 0.0
    delta = (mae_random - mae) / sd_random if sd_random != 0 else 0.0
    
    return mae, sa, delta

from joblib import Parallel, delayed

def run_loocv(dataset_path, use_knn_imputation=False):
    print(f"\n--- Running LOOCV for {os.path.basename(dataset_path)} (KNN={use_knn_imputation}) ---", flush=True)
    
    target_col = 'Effort' if 'cocomo' not in dataset_path.lower() else 'actual'
    ds_full = Dataset(dataset_path, target_column_name=target_col, use_knn_imputation=use_knn_imputation)
    n_samples = len(ds_full.data)
    
    is_china = 'china' in dataset_path.lower()
    if is_china:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        print("NOTE: Using 10-fold CV (not LOOCV) for China as a preliminary check.", flush=True)
    else:
        cv = LeaveOneOut()
        
    def process_single_fold(train_idx, test_idx):
        import torch
        import random
        torch.set_num_threads(1)
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        # Determine device
        is_china = 'china' in dataset_path.lower()
        if is_china and torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            device = 'cpu'
        
        dataset = Dataset(dataset_path, target_column_name=target_col, train_indices=train_idx, test_indices=test_idx, use_knn_imputation=use_knn_imputation)
        fuzzy_model, optimal_k, mu_train = elbow_XB(dataset, max_clusters=10)
        mu_test = fuzzy_model.transform(dataset.testing_data)
        
        t_mean = dataset.training_target_unscaled.mean()
        estimator = EffortEstimator(
            m_features=dataset.training_data.shape[0],
            k_clusters=optimal_k,
            lr=0.001,
            epochs=400,
            batch_size=16,
            target_min=0.0,
            target_range=1.0,
            target_mean=t_mean,
            device=device
        )
        
        estimator.fit(mu_train, dataset.training_target_unscaled.values)
        pred_unscaled = estimator.predict(mu_test)
        
        actuals_fold = dataset.testing_target_unscaled.values
        predictions_fold = pred_unscaled
        return actuals_fold, predictions_fold

    # Use joblib for parallel execution
    is_china = 'china' in dataset_path.lower()
    # Use n_jobs=1 for GPU to avoid extreme context switching overhead and VRAM OOM
    # or n_jobs=-1 for CPU
    import torch
    num_jobs = 1 if (is_china and torch.cuda.is_available()) else -1
    
    results = Parallel(n_jobs=num_jobs, verbose=5)(
        delayed(process_single_fold)(train_idx, test_idx) for train_idx, test_idx in cv.split(ds_full.data)
    )
    
    actuals = []
    predictions = []
    fold_indices = []
    
    for idx, res in enumerate(results):
        actuals.extend(res[0])
        predictions.extend(res[1])
        fold_indices.extend([idx] * len(res[0]))
        
    mae, sa, delta = compute_metrics(actuals, predictions)
    
    # Save per-fold results
    df_results = pd.DataFrame({
        'fold_index': fold_indices,
        'actual': actuals,
        'predicted': predictions
    })
    df_results['abs_error'] = np.abs(df_results['actual'] - df_results['predicted'])
    df_results['squared_error'] = df_results['abs_error'] ** 2
    ds_name_base = os.path.basename(dataset_path)
    csv_filename = f"results_per_fold_{ds_name_base}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"Saved per-fold results to {csv_filename}")
    
    print(f"Results for {os.path.basename(dataset_path)}:")
    print(f"MAE: {mae:.4f}")
    print(f"SA:  {sa:.4f}")
    print(f"Delta: {delta:.4f}")
    
    return mae, sa, delta
    
    print(f"Results for {os.path.basename(dataset_path)}:")
    print(f"MAE: {mae:.4f}")
    print(f"SA:  {sa:.4f}")
    print(f"Delta: {delta:.4f}")
    
    return mae, sa, delta
    
    print(f"Results for {os.path.basename(dataset_path)}:")
    print(f"MAE: {mae:.4f}")
    print(f"SA:  {sa:.4f}")
    print(f"Delta: {delta:.4f}")
    
    return mae, sa, delta

if __name__ == '__main__':
    datasets = [
        '../Software-estimation-datasets/COCOMO-81.csv',
    ]
    
    import time
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print("Note: Using GPU ONLY for China.arff. CPU is faster for other small datasets due to transfer overhead.")
    else:
        print("No GPU detected.")
    
    results = {}
    for ds_path in datasets:
        if os.path.exists(ds_path):
            start_time = time.time()
            mae, sa, delta = run_loocv(ds_path)
            elapsed = time.time() - start_time
            method = '10-fold' if 'china' in ds_path.lower() else 'LOOCV'
            results[os.path.basename(ds_path)] = {'MAE': mae, 'SA': sa, 'Delta': delta, 'Time': elapsed, 'Method': method}
            print(f"Time for {os.path.basename(ds_path)}: {elapsed:.2f} seconds")
        else:
            print(f"File not found: {ds_path}")
            
    print("\n" + "="*75)
    print("FINAL SUMMARY (POST-FIX LOOCV / 10-FOLD)")
    print("="*75)
    print(f"{'Dataset':<15} | {'Method':<10} | {'MAE':<10} | {'SA':<10} | {'Delta':<10} | {'Time (s)':<10}")
    for k, v in results.items():
        print(f"{k:<15} | {v['Method']:<10} | {v['MAE']:<10.2f} | {v['SA']:<10.2f} | {v['Delta']:<10.2f} | {v['Time']:<10.2f}")
    print("="*75)
