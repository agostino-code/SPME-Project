"""Profile ONE outer fold of nested CV on China (n=499)."""
import cProfile
import pstats
import io
import time
import numpy as np
from sklearn.model_selection import KFold
from dataset import Dataset
from nested_cv import optimize_hyperparameters
from fuzzy import FuzzyCMeans
from DCNN import EffortEstimator

def profile_one_fold():
    dataset_path = "../Software-estimation-datasets/china.arff"
    target_column = "Effort"
    
    full_ds = Dataset(dataset_path, target_column_name=target_column)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(outer_cv.split(full_ds.data))
    
    train_idx, test_idx = splits[0]
    
    print(f"China: {len(full_ds.data)} samples total")
    print(f"Fold 0: train={len(train_idx)}, test={len(test_idx)}")
    
    t0 = time.perf_counter()
    
    best_params = optimize_hyperparameters(
        dataset_path=dataset_path,
        target_column=target_column,
        outer_train_indices=train_idx,
        n_inner_trials=10
    )
    
    dataset = Dataset(
        dataset_path,
        target_column_name=target_column,
        test_indices=list(test_idx),
        top_k_features=best_params.get('top_k_features'),
        apply_skew_transform=True,
        use_pca=best_params.get('use_pca', False),
        augment_rare_targets=best_params.get('augment_rare_targets', False)
    )
    
    n_clusters = best_params.get('num_clusters', 3)
    actual_clusters = min(n_clusters, max(2, dataset.training_data.shape[1] - 1))
    
    fuzzy_model = FuzzyCMeans(
        dataset, num_clusters=actual_clusters,
        m=best_params.get('m', 2.0),
        distance_metric=best_params.get('distance_metric', 'euclidean')
    )
    mu_train = fuzzy_model.fit_transform()
    mu_test = fuzzy_model.transform(dataset.testing_data)
    
    estimator = EffortEstimator(
        n_features=mu_train.shape[1], num_clusters=actual_clusters,
        lr=best_params.get('lr', 1e-3), batch_size=16, epochs=150,
        weight_decay=best_params.get('weight_decay', 1e-3),
        kernels=best_params.get('kernels', 16),
        target_min=dataset.target_min, target_range=dataset.target_range,
        loss_type=best_params.get('loss_type', 'hybrid'),
        alpha=best_params.get('alpha', 0.5),
        use_log_target=dataset.use_log_target
    )
    estimator.fit(mu_train, dataset.training_target_scaled,
                  X_train_raw=dataset.training_data, verbose=False,
                  patience=20, val_ratio=0.15)
    
    mae, rmse, mdae, mmre, mdmre, pred25 = estimator.evaluate(
        mu_test, dataset.testing_target_scaled,
        X_test_raw=dataset.testing_data,
        inverse_transform=dataset.inverse_transform_target
    )
    
    elapsed = time.perf_counter() - t0
    print(f"\nWall-clock for 1 outer fold: {elapsed:.1f}s")
    print(f"Metrics: MAE={mae:.2f} RMSE={rmse:.2f} MMRE={mmre:.4f} Pred25={pred25:.1f}%")
    print(f"Best params: {best_params}")

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    profile_one_fold()
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print("\n=== TOP-30 BY CUMULATIVE TIME ===")
    print(s.getvalue())
