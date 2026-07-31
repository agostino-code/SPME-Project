import os
import time
import numpy as np
import pandas as pd
import torch
import warnings
import itertools
from sklearn.model_selection import KFold
from dataset import Dataset
from elbow import fixed_fcm
from DCNN import EffortEstimator
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

def compute_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def run_fold(fold_idx, train_idx, test_idx, dataset_path, target_col, model_type, num_filters, dropout_rate, force_k, fuzzifier_m):
    dataset = Dataset(dataset_path, target_column_name=target_col, train_indices=train_idx, test_indices=test_idx)
    
    # Ensures deterministic fold
    torch.manual_seed(42 + fold_idx)
    np.random.seed(42 + fold_idx)
    
    # Generate Fuzzy Space
    fuzzy_model, k_clusters, mu_train = fixed_fcm(dataset, force_k, fuzzifier_m)
    mu_test = fuzzy_model.transform(dataset.testing_data)
    
    # Original target settings
    t_min = dataset.training_target_unscaled.min()
    t_max = dataset.training_target_unscaled.max()
    t_range = t_max - t_min if t_max != t_min else 1.0
    target_mean = dataset.training_target_unscaled.mean()
    
    estimator = EffortEstimator(
        m_features=dataset.training_data.shape[0], 
        k_clusters=k_clusters,
        lr=0.001, epochs=150, batch_size=16, 
        target_min=t_min, target_range=t_range, target_mean=target_mean, 
        device='cpu', 
        model_type=model_type, 
        num_filters=num_filters, 
        dropout_rate=dropout_rate
    )
    
    estimator.fit(mu_train, dataset.training_target_unscaled.values)
    preds = estimator.predict(mu_test)
    
    return dataset.testing_target_unscaled.values, preds

def evaluate_pipeline_config(dataset_path, model_type, num_filters, dropout_rate, force_k, fuzzifier_m):
    target_col = 'Effort' if 'cocomo' not in dataset_path.lower() else 'actual'
    ds_full = Dataset(dataset_path, target_column_name=target_col)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(ds_full.data))
    
    start_time = time.time()
    
    results = Parallel(n_jobs=-1)(
        delayed(run_fold)(i, tr, te, dataset_path, target_col, model_type, num_filters, dropout_rate, force_k, fuzzifier_m)
        for i, (tr, te) in enumerate(splits)
    )
    
    exec_time = time.time() - start_time
    
    all_actual = []
    all_preds = []
    for actual, preds in results:
        all_actual.extend(actual)
        all_preds.extend(preds)
        
    mae = compute_mae(all_actual, all_preds)
    return mae, exec_time

def tune():
    datasets = [
        '../Software-estimation-datasets/china.arff',
        '../Software-estimation-datasets/Desharnais.csv'
    ]
    
    architectures = [
        ('FDCNN', 32, 0.2),
        ('FDCNN_SeqAsym', 32, 0.5)
    ]
    
    k_vals = [3, 5, 7, 10]
    m_vals = [1.5, 2.0, 2.5]
    
    grid = list(itertools.product(architectures, k_vals, m_vals))
    
    results_df = []
    
    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        print(f"\n{'='*60}\nStarting Pipeline Grid Search on {ds_name}\n{'='*60}")
        
        best_mae = float('inf')
        best_config = None
        
        for arch, force_k, fuzzifier_m in grid:
            model_type, num_f, drop = arch
            print(f"[{model_type}] K={force_k:2d} | M={fuzzifier_m:.1f} ... ", end='', flush=True)
            
            mae, exec_time = evaluate_pipeline_config(ds_path, model_type, num_f, drop, force_k, fuzzifier_m)
            print(f"MAE = {mae:.2f} (Time: {exec_time:.1f}s)")
            
            results_df.append({
                'Dataset': ds_name,
                'Model': model_type,
                'Filters': num_f,
                'Dropout': drop,
                'Force_K': force_k,
                'Fuzzifier_M': fuzzifier_m,
                'MAE': mae,
                'Time_Seconds': exec_time
            })
            
            if mae < best_mae:
                best_mae = mae
                best_config = (model_type, force_k, fuzzifier_m)
                
        print(f"\n>>> BEST FCM CONFIG FOR {ds_name}:")
        print(f"Model: {best_config[0]} | K: {best_config[1]} | M: {best_config[2]} | MAE: {best_mae:.2f}")
        
    df = pd.DataFrame(results_df)
    df.to_csv('fcm_tuning_results.csv', index=False)
    print("\nResults saved to fcm_tuning_results.csv")

if __name__ == "__main__":
    tune()
