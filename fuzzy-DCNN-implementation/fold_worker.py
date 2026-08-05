import random
import numpy as np
import torch

from dataset import Dataset
from elbow import *
from DCNN import EffortEstimator


def is_china_dataset(dataset_path: str) -> bool:
    """China.arff è l'unico dataset abbastanza grande da giustificare 10-fold CV invece di LOOCV."""
    return 'china' in dataset_path.lower()


def target_column_for(dataset_path: str) -> str:
    """COCOMO usa 'actual' come nome della colonna target; tutti gli altri usano 'Effort'."""
    return 'actual' if 'cocomo' in dataset_path.lower() else 'Effort'


def process_single_fold(dataset_path: str, target_col: str, use_knn_imputation: bool,
                        train_idx: np.ndarray, test_idx: np.ndarray,
                        random_seed: int = 42,
                        max_fuzzy_clusters: int = 10,
                        dcnn_lr: float = 0.001, dcnn_epochs: int = 1000, dcnn_batch_size: int = 16,
                        num_filters: int = 16, dropout_rate: float = 0.2, weight_decay: float = 0.0,
                        loss_type: str = 'huber', huber_delta: float = 0.01,
                        activation_type: str = 'linear_clip',
                        fuzzy_m: float = 2.0,
                        use_log_target: bool = False,
                        cluster_method: str | int = 'XB-index'):
    """
    Allena e valuta il modello Fuzzy-DCNN su un singolo fold LOOCV/K-fold.
    """
    torch.set_num_threads(1)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    use_gpu = is_china_dataset(dataset_path) and torch.cuda.is_available()
    if use_gpu:
        device = 'cuda'
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    else:
        device = 'cpu'

    dataset = Dataset(
        dataset_path,
        target_column_name=target_col,
        train_indices=train_idx,
        test_indices=test_idx,
        use_knn_imputation=use_knn_imputation,
        use_log_target=use_log_target,
    )

    if cluster_method == 'XB-index':
        fuzzy_model, optimal_k, mu_train = elbow_XB(dataset, max_clusters=max_fuzzy_clusters, fuzzifier_m=fuzzy_m)
    elif cluster_method == 'silhouette':
        fuzzy_model, optimal_k, mu_train = elbow_silhouette(dataset, max_clusters=max_fuzzy_clusters, fuzzifier_m=fuzzy_m)
    else:  # int (fixed method)
        fuzzy_model, optimal_k, mu_train = fixed_fcm(dataset, cluster_method, fuzzifier_m=fuzzy_m)
    mu_test = fuzzy_model.transform(dataset.testing_data)

    target_mean_scaled = dataset.training_target_scaled.mean()
        
    estimator = EffortEstimator(
        m_features=dataset.training_data.shape[0],
        k_clusters=optimal_k,
        lr=dcnn_lr,
        epochs=dcnn_epochs,
        batch_size=dcnn_batch_size,
        weight_decay=weight_decay,
        target_min=0.0,
        target_range=1.0,
        target_mean=target_mean_scaled,
        device=device,
        num_filters=num_filters,
        dropout_rate=dropout_rate,
        loss_type=loss_type,
        huber_delta=huber_delta,
        activation_type=activation_type,
    )
    estimator.fit(mu_train, dataset.training_target_scaled.values)

    pred_scaled = estimator.predict(mu_test)
    pred_unscaled = dataset.inverse_transform_target(pred_scaled)

    actuals_fold = dataset.testing_target_unscaled.values
    predictions_fold = pred_unscaled
    return actuals_fold, predictions_fold
