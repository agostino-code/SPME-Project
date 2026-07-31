import os
import numpy as np
import pandas as pd
import torch
import warnings
import itertools
from sklearn.model_selection import KFold
from dataset import Dataset
from elbow import elbow_XB
from DCNN import EffortEstimator
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

def compute_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def run_fold(fold_idx, train_idx, test_idx, dataset_path, target_col, model_type, num_filters, dropout_rate):
    dataset = Dataset(dataset_path, target_column_name=target_col, train_indices=train_idx, test_indices=test_idx)
    
    # Ensures deterministic fold
    torch.manual_seed(42 + fold_idx)
    np.random.seed(42 + fold_idx)
    
    fuzzy_model, optimal_k, mu_train = elbow_XB(dataset, max_clusters=10)
    mu_test = fuzzy_model.transform(dataset.testing_data)
    
    # We will use original scaling as baseline behavior (target_mean logic)
    t_min = dataset.training_target_unscaled.min()
    t_max = dataset.training_target_unscaled.max()
    t_range = t_max - t_min if t_max != t_min else 1.0
    
    # In Phase 1 we established to use original target mean bias to start safely
    target_mean = dataset.training_target_unscaled.mean()
    
    estimator = EffortEstimator(
        m_features=dataset.training_data.shape[0], 
        k_clusters=optimal_k,
        lr=0.001, epochs=150, batch_size=16, 
        target_min=t_min, target_range=t_range, target_mean=target_mean, 
        device='cpu', 
        model_type=model_type, 
        num_filters=num_filters, 
        dropout_rate=dropout_rate
    )
    
    # We use original unscaled targets for simple evaluation (since scaling failed)
    # Actually wait! The MSE loss gradient exploded on COCOMO. But for China/Desharnais it's okay.
    # To be exactly faithful to our Phase 1 baseline F-DCNN:
    estimator.fit(mu_train, dataset.training_target_unscaled.values)
    preds = estimator.predict(mu_test)
    
    return dataset.testing_target_unscaled.values, preds

def evaluate_config(dataset_path, model_type, num_filters, dropout_rate):
    target_col = 'Effort' if 'cocomo' not in dataset_path.lower() else 'actual'
    ds_full = Dataset(dataset_path, target_column_name=target_col)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(ds_full.data))
    
    results = Parallel(n_jobs=-1)(
        delayed(run_fold)(i, tr, te, dataset_path, target_col, model_type, num_filters, dropout_rate)
        for i, (tr, te) in enumerate(splits)
    )
    
    all_actual = []
    all_preds = []
    for actual, preds in results:
        all_actual.extend(actual)
        all_preds.extend(preds)
        
    mae = compute_mae(all_actual, all_preds)
    return mae

def tune():
    datasets = [
        '../Software-estimation-datasets/china.arff',
        '../Software-estimation-datasets/Desharnais.csv'
    ]
    
    models = ['FDCNN', 'FDCNN_SeqAsym', 'FDCNN_ParAsym']
    filters = [8, 16, 32]
    dropouts = [0.1, 0.2, 0.5]
    
    grid = list(itertools.product(models, filters, dropouts))
    
    results_df = []
    
    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        print(f"\n{'='*50}\nStarting Grid Search on {ds_name}\n{'='*50}")
        
        best_mae = float('inf')
        best_config = None
        
        for model_type, num_f, drop in grid:
            print(f"Testing {model_type} | Filters: {num_f:2d} | Dropout: {drop:.1f} ... ", end='', flush=True)
            mae = evaluate_config(ds_path, model_type, num_f, drop)
            print(f"MAE = {mae:.2f}")
            
            results_df.append({
                'Dataset': ds_name,
                'Model': model_type,
                'Filters': num_f,
                'Dropout': drop,
                'MAE': mae
            })
            
            if mae < best_mae:
                best_mae = mae
                best_config = (model_type, num_f, drop)
                
        print(f"\n>>> BEST CONFIG FOR {ds_name}:")
        print(f"Model: {best_config[0]} | Filters: {best_config[1]} | Dropout: {best_config[2]} | MAE: {best_mae:.2f}")
        
    df = pd.DataFrame(results_df)
    df.to_csv('tuning_results.csv', index=False)
    print("\nResults saved to tuning_results.csv")

if __name__ == "__main__":
    tune()
