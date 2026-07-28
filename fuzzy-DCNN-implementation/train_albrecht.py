import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from fuzzy import FuzzyCMeans
from dataset import Dataset
from DCNN import EffortEstimator
from elbow import elbow_XB


class Config:
    def __init__(self):
        self.max_clusters = 4
        self.fixed_clusters = True
        self.epochs = 100
        self.weight_decay = 5e-3
        self.batch_size = 1
        self.lr = 5e-3
        self.kernels = 8
        self.n_folds = 10
        self.LOO = True

    def __str__(self):
        return str(self.__dict__)


def parse_args():
    return Config()


def train_and_evaluate(test_indices, args):
    dataset_path = "../Software-estimation-datasets/albrecht.arff"
    target_column = "Effort"
    dataset = Dataset(
        dataset_path, target_column_name=target_column, test_indices=test_indices
    )

    if args.fixed_clusters:
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=args.max_clusters)
        mu = fuzzy_cmeans.fit_transform()
        n_clusters = args.max_clusters
    else:
        fuzzy_cmeans, n_clusters, mu = elbow_XB(dataset, max_clusters=args.max_clusters)
        print(
            f"[*] Elbow method selected {n_clusters} clusters (max tested was {int(np.sqrt(dataset.training_data.shape[1]))})"
        )

    n_features = mu.shape[1]
    effort_estimator = EffortEstimator(
        n_features,
        n_clusters,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        kernels=args.kernels,
        target_min=dataset.target_min,
        target_range=dataset.target_range,
    )
    effort_estimator.fit(mu, dataset.training_target_scaled, verbose=False)

    test_mae, test_rmse, test_mdae, test_mmre, test_mdmre, test_pred25 = (
        effort_estimator.evaluate(
            fuzzy_cmeans.transform(dataset.testing_data),
            dataset.testing_target_scaled,
            inverse_transform=dataset.inverse_transform_target,
        )
    )

    return test_mae, test_rmse, test_mdae, test_mmre, test_mdmre, test_pred25


if __name__ == "__main__":
    args = parse_args()

    avg_test_mae = 0.0
    avg_test_rmse = 0.0
    avg_test_mdae = 0.0
    avg_test_mmre = 0.0
    avg_test_mdmre = 0.0
    avg_test_pred25 = 0.0
    min_test_mae = float("inf")
    max_test_mae = float("-inf")

    # Load just to get length
    full_dataset = Dataset(
        "../Software-estimation-datasets/albrecht.arff", target_column_name="Effort"
    )

    print("=" * 60)
    print(f"TRAINING on Albrecht dataset with the following configuration:")
    print(f"\n")
    print(f"Max Clusters: {args.max_clusters}")
    print(f"Fixed Clusters: {args.fixed_clusters}")
    print(f"Epochs: {args.epochs}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Kernels: {args.kernels}")
    print(f"Number of Folds: {args.n_folds}")
    print(f"Leave-One-Out: {args.LOO}")
    print("=" * 60)

    if args.LOO:
        loo = LeaveOneOut()
        splits = list(loo.split(full_dataset.data))
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        splits = list(kf.split(full_dataset.data))

    n_splits = len(splits)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Running iteration {fold_idx + 1}/{n_splits}")
        mae, rmse, mdae, mmre, mdmre, pred25 = train_and_evaluate(test_idx, args)

        print(
            f"Iteration {fold_idx + 1} MAE: {mae:.4f} RMSE: {rmse:.4f} MMRE: {mmre:.4f} Pred(25): {pred25:.2f}%"
        )

        avg_test_mae += mae
        avg_test_rmse += rmse
        avg_test_mdae += mdae
        avg_test_mmre += mmre
        avg_test_mdmre += mdmre
        avg_test_pred25 += pred25
        min_test_mae = min(min_test_mae, mae)
        max_test_mae = max(max_test_mae, mae)

    avg_test_mae /= n_splits
    avg_test_rmse /= n_splits
    avg_test_mdae /= n_splits
    avg_test_mmre /= n_splits
    avg_test_mdmre /= n_splits
    avg_test_pred25 /= n_splits

    print("=" * 60)
    print(f"Average Test MAE:     {avg_test_mae:.4f}")
    print(f"Average Test RMSE:    {avg_test_rmse:.4f}")
    print(f"Average Test MdAE:    {avg_test_mdae:.4f}")
    print(f"Average Test MMRE:    {avg_test_mmre:.4f}")
    print(f"Average Test MdMRE:   {avg_test_mdmre:.4f}")
    print(f"Average Test Pred(25):{avg_test_pred25:.2f}%")
    print(f"Minimum Test MAE:     {min_test_mae:.4f}")
    print(f"Maximum Test MAE:     {max_test_mae:.4f}")
