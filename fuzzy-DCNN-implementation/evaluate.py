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
    delta = (mae - mae_random) / sd_random if sd_random != 0 else 0.0
    
    return mae, sa, delta

from joblib import Parallel, delayed

def run_loocv(dataset_path):
    print(f"\n--- Running LOOCV for {os.path.basename(dataset_path)} ---", flush=True)
    
    target_col = 'Effort' if 'cocomo' not in dataset_path.lower() else 'actual'
    ds_full = Dataset(dataset_path, target_column_name=target_col)
    n_samples = len(ds_full.data)
    
    loo = LeaveOneOut()
    
    def process_single_fold(train_idx, test_idx):
        import torch
        torch.set_num_threads(1)
        dataset = Dataset(dataset_path, target_column_name=target_col, train_indices=train_idx, test_indices=test_idx)
        fuzzy_model, optimal_k, mu_train = elbow_XB(dataset, max_clusters=10)
        mu_test = fuzzy_model.transform(dataset.testing_data)
        
        estimator = EffortEstimator(
            m_features=dataset.training_data.shape[0],
            k_clusters=optimal_k,
            lr=0.001,
            epochs=150,
            batch_size=16,
            target_min=0.0,
            target_range=1.0
        )
        
        estimator.fit(mu_train, dataset.training_target_unscaled.values)
        pred_unscaled = estimator.predict(mu_test)
        
        actual = dataset.testing_target_unscaled.values[0]
        predicted = pred_unscaled[0]
        return actual, predicted

    # Use joblib for parallel execution
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_single_fold)(train_idx, test_idx) for train_idx, test_idx in loo.split(ds_full.data)
    )
    
    actuals = [res[0] for res in results]
    predictions = [res[1] for res in results]
    
    mae, sa, delta = compute_metrics(actuals, predictions)
    
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
        '../Software-estimation-datasets/albrecht.arff',
        '../Software-estimation-datasets/china.arff',
        '../Software-estimation-datasets/maxwell.arff',
        '../Software-estimation-datasets/Desharnais.csv',
        '../Software-estimation-datasets/COCOMO-81.csv'
    ]
    
    results = {}
    for ds_path in datasets:
        if os.path.exists(ds_path):
            mae, sa, delta = run_loocv(ds_path)
            results[os.path.basename(ds_path)] = {'MAE': mae, 'SA': sa, 'Delta': delta}
        else:
            print(f"File not found: {ds_path}")
            
    print("\n" + "="*50)
    print("FINAL SUMMARY (BASELINE LOOCV)")
    print("="*50)
    print(f"{'Dataset':<15} | {'MAE':<10} | {'SA':<10} | {'Delta':<10}")
    for k, v in results.items():
        print(f"{k:<15} | {v['MAE']:<10.2f} | {v['SA']:<10.2f} | {v['Delta']:<10.2f}")
    print("="*50)
