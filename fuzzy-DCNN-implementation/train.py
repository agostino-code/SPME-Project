import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from fuzzy import FuzzyCMeans
from dataset import Dataset
from DCNN import EffortEstimator

class Config:
    def __init__(self):
        self.max_clusters = 10
        self.fixed_clusters = False
        self.epochs = 150
        self.weight_decay = 1e-3
        self.batch_size = 16
        self.lr = 1e-3
        self.LOO = False

    def __str__(self):
        return str(self.__dict__)

def parse_args():
    return Config()

def elbow_XB(dataset, max_clusters):
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
    dataset = Dataset(dataset_path, target_column_name=target_column, test_indices=test_indices)
    
    if args.fixed_clusters:
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=args.max_clusters)
        mu = fuzzy_cmeans.fit_transform()
        n_clusters = args.max_clusters
    else:
        fuzzy_cmeans, n_clusters, mu = elbow_XB(dataset, max_clusters=args.max_clusters)
        print(f"[*] Elbow method selected {n_clusters} clusters (max tested was {int(np.sqrt(dataset.training_data.shape[1]))})")

    n_features = mu.shape[1]
    effort_estimator = EffortEstimator(
        n_features, n_clusters, 
        lr=args.lr, 
        batch_size=args.batch_size, 
        epochs=args.epochs,
        weight_decay=args.weight_decay,
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
    full_dataset = Dataset('../Software-estimation-datasets/china.arff', target_column_name='Effort')
    
    # --- INIZIO MODIFICA LOOCV ---
    
    if args.LOO:
        loo = LeaveOneOut()
        splits = list(loo.split(full_dataset.data))
    else:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        splits = list(kf.split(full_dataset.data))
    
    n_splits = len(splits)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Running iteration {fold_idx + 1}/{n_splits}")
        test_mae, test_rmse, test_mdae = train_and_evaluate(test_idx, args)
        
        # Stampa l'errore per il singolo sample (opzionale, utile per debug)
        print(f"Iteration {fold_idx + 1} MAE: {test_mae:.4f} RMSE: {test_rmse:.4f}")
        
        avg_test_mae += test_mae
        avg_test_rmse += test_rmse
        avg_test_mdae += test_mdae
        min_test_mae = min(min_test_mae, test_mae)
        max_test_mae = max(max_test_mae, test_mae)

    avg_test_mae /= n_splits
    avg_test_rmse /= n_splits
    avg_test_mdae /= n_splits

    print("=" * 40)
    print(f"Average Test MAE: {avg_test_mae:.4f}")
    print(f"Average Test RMSE: {avg_test_rmse:.4f}")
    print(f"Average Test MdAE: {avg_test_mdae:.4f}")
    print(f"Minimum Test MAE: {min_test_mae:.4f}")
    print(f"Maximum Test MAE: {max_test_mae:.4f}")