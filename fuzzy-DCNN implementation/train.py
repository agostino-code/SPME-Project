import argparse
import numpy as np
from sklearn.model_selection import KFold
from fuzzy import FuzzyCMeans
from dataset import Dataset
from DCNN import EffortEstimator

def parse_args():
    parser = argparse.ArgumentParser(description="Train F-DCNN for Effort Estimation on China Dataset")
    parser.add_argument('--features', type=str, nargs='+', default=None, help='List of selected features (e.g. HLFPA IFPA AFP)')
    parser.add_argument('--clusters', type=int, default=10, help='Number of max clusters for elbow XB, or exact number if fixed')
    parser.add_argument('--fixed_clusters', action='store_true', help='Use fixed number of clusters instead of elbow method')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--kernels', type=int, default=16)
    parser.add_argument('--hidden', type=int, nargs='+', default=[32, 16])
    return parser.parse_args()

def elbow_XB(dataset, max_clusters=10):
    n_samples = dataset.training_data.shape[1]
    max_clusters = min(max_clusters, max(2, int(np.sqrt(n_samples))))
    
    current_best_XB = float('inf')
    current_best_fuzzy_cmeans = None
    current_best_mu = None
    current_best_cluster_n = 0
    
    for n_clusters in range(2, max_clusters + 1):
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=n_clusters)
        mu = fuzzy_cmeans.fit_transform()
        
        current_XB = fuzzy_cmeans.XB()
        if current_XB < current_best_XB:
            current_best_XB = current_XB
            current_best_fuzzy_cmeans = fuzzy_cmeans
            current_best_mu = mu
            current_best_cluster_n = n_clusters

    assert current_best_fuzzy_cmeans is not None and current_best_mu is not None and current_best_cluster_n > 0, "No valid fuzzy c-means model found."

    return current_best_fuzzy_cmeans, current_best_cluster_n, current_best_mu

def train_and_evaluate(test_indices, args):
    dataset_path = '../Software-estimation-datasets/china.arff'
    target_column = 'Effort'
    dataset = Dataset(dataset_path, target_column_name=target_column, test_indices=test_indices, selected_features=args.features)
    
    if args.fixed_clusters:
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=args.clusters)
        mu = fuzzy_cmeans.fit_transform()
        n_clusters = args.clusters
    else:
        fuzzy_cmeans, n_clusters, mu = elbow_XB(dataset, max_clusters=args.clusters)

    n_features = mu.shape[1]
    effort_estimator = EffortEstimator(
        n_features, n_clusters, 
        lr=args.lr, 
        batch_size=args.batch_size, 
        epochs=args.epochs,
        kernels=args.kernels,
        hidden_layers=args.hidden,
        dropout_prob=args.dropout,
        target_min=dataset.target_min,
        target_range=dataset.target_range
    )
    effort_estimator.fit(mu, dataset.training_target_scaled, verbose=True)
    
    test_mae, test_rmse, test_mdae = effort_estimator.evaluate(
        fuzzy_cmeans.transform(dataset.testing_data),
        dataset.testing_target_scaled,
        inverse_transform=dataset.inverse_transform_target,
    )
    
    return test_mae, test_rmse, test_mdae

if __name__ == "__main__":
    args = parse_args()

    avg_test_mae = 0
    avg_test_rmse = 0
    avg_test_mdae = 0
    min_test_mae = float('inf')
    max_test_mae = float('-inf')

    # Load just to get length
    full_dataset = Dataset('../Software-estimation-datasets/china.arff', target_column_name='Effort', selected_features=args.features)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(kf.split(full_dataset.data))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Running fold {fold_idx + 1}/10")
        test_mae, test_rmse, test_mdae = train_and_evaluate(test_idx, args)
        print(f"Fold {fold_idx + 1} MAE: {test_mae:.4f} RMSE: {test_rmse:.4f} MdAE: {test_mdae:.4f}")
        avg_test_mae += test_mae
        avg_test_rmse += test_rmse
        avg_test_mdae += test_mdae
        min_test_mae = min(min_test_mae, test_mae)
        max_test_mae = max(max_test_mae, test_mae)

    avg_test_mae /= 10
    avg_test_rmse /= 10
    avg_test_mdae /= 10

    print("=" * 40)
    print("Configuration:")
    print(args)
    print("=" * 40)
    print(f"Average Test MAE: {avg_test_mae:.4f}")
    print(f"Average Test RMSE: {avg_test_rmse:.4f}")
    print(f"Average Test MdAE: {avg_test_mdae:.4f}")
    print(f"Minimum Test MAE: {min_test_mae:.4f}")
    print(f"Maximum Test MAE: {max_test_mae:.4f}")