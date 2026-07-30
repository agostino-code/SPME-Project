import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold, LeaveOneOut
from dataset import Dataset
from fuzzy import FuzzyCMeans
from DCNN import EffortEstimator

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hyperparameters(dataset_path, target_column, outer_train_indices, n_inner_trials=20, n_inner_folds=3):
    """
    Inner loop of Nested CV: Uses Optuna to tune hyperparameters strictly on outer training data.
    """
    def objective(trial):
        m = trial.suggest_float('m', 1.1, 3.0)
        n_clusters = trial.suggest_int('num_clusters', 2, 5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        kernels = trial.suggest_categorical('kernels', [8, 16, 32])
        loss_type = trial.suggest_categorical('loss_type', ['hybrid', 'mre', 'mae', 'huber'])
        alpha = trial.suggest_float('alpha', 0.1, 0.9)
        distance_metric = trial.suggest_categorical('distance_metric', ['euclidean', 'mahalanobis'])
        top_k_features = trial.suggest_categorical('top_k_features', [None, 4, 6, 8, 10])
        use_pca = trial.suggest_categorical('use_pca', [True, False])
        augment_rare_targets = trial.suggest_categorical('augment_rare_targets', [True, False])

        dataset = Dataset(
            dataset_path,
            target_column_name=target_column,
            test_indices=None,
            top_k_features=top_k_features,
            apply_skew_transform=True,
            use_pca=use_pca,
            augment_rare_targets=augment_rare_targets
        )

        n_samples = dataset.training_data.shape[1]
        if n_samples < 5:
            return float('inf')

        kf_inner = KFold(n_splits=min(n_inner_folds, n_samples), shuffle=True, random_state=42)
        inner_scores = []

        for in_tr_idx, in_val_idx in kf_inner.split(dataset.training_features_df):
            in_test_indices = list(in_val_idx)

            sub_dataset = Dataset(
                dataset_path,
                target_column_name=target_column,
                test_indices=in_test_indices,
                top_k_features=top_k_features,
                apply_skew_transform=True,
                use_pca=use_pca,
                augment_rare_targets=augment_rare_targets
            )


            actual_clusters = min(n_clusters, max(2, sub_dataset.training_data.shape[1] - 1))
            try:
                fuzzy_model = FuzzyCMeans(
                    sub_dataset,
                    num_clusters=actual_clusters,
                    m=m,
                    distance_metric=distance_metric
                )
                mu_train = fuzzy_model.fit_transform()
                mu_val = fuzzy_model.transform(sub_dataset.testing_data)

                estimator = EffortEstimator(
                    n_features=mu_train.shape[1],
                    num_clusters=actual_clusters,
                    lr=lr,
                    batch_size=16,
                    epochs=100,
                    weight_decay=weight_decay,
                    kernels=kernels,
                    target_min=sub_dataset.target_min,
                    target_range=sub_dataset.target_range,
                    loss_type=loss_type,
                    alpha=alpha,
                    use_log_target=sub_dataset.use_log_target
                )
                estimator.fit(
                    mu_train,
                    sub_dataset.training_target_scaled,
                    X_train_raw=sub_dataset.training_data,
                    verbose=False,
                    patience=15,
                    val_ratio=0.15
                )

                mae, _, _, mmre, _, _ = estimator.evaluate(
                    mu_val,
                    sub_dataset.testing_target_scaled,
                    X_test_raw=sub_dataset.testing_data,
                    inverse_transform=sub_dataset.inverse_transform_target
                )
                
                # Balanced Multi-Objective Score: Normalized MAE + MMRE
                mean_target = sub_dataset.inverse_transform_target(np.mean(sub_dataset.training_target_scaled))
                mae_norm = mae / max(1.0, mean_target)
                score = 0.5 * mae_norm + 0.5 * mmre
                inner_scores.append(score)
            except Exception:
                return float('inf')

        return float(np.mean(inner_scores)) if inner_scores else float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_inner_trials)
    return study.best_params

def run_nested_cv(dataset_path, target_column='Effort', n_outer_folds=5, n_trials=15, is_loo=False):
    """
    Executes Nested Cross-Validation (Outer CV for evaluation, Inner Optuna CV for hyperparameter tuning).
    """
    print("=" * 70)
    print(f"STARTING NESTED CROSS-VALIDATION on: {dataset_path}")
    print(f"Outer Folds: {n_outer_folds if not is_loo else 'LOOCV'} | Inner Optuna Trials: {n_trials}")
    print("=" * 70)

    full_ds = Dataset(dataset_path, target_column_name=target_column)

    if is_loo:
        outer_cv = LeaveOneOut()
        splits = list(outer_cv.split(full_ds.data))
    else:
        outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
        splits = list(outer_cv.split(full_ds.data))

    outer_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n--- Outer Fold {fold_idx + 1}/{len(splits)} ---")

        best_params = optimize_hyperparameters(
            dataset_path=dataset_path,
            target_column=target_column,
            outer_train_indices=test_idx,
            n_inner_trials=n_trials
        )
        print(f"Best Hyperparams for Fold {fold_idx + 1}: {best_params}")

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
            dataset,
            num_clusters=actual_clusters,
            m=best_params.get('m', 2.0),
            distance_metric=best_params.get('distance_metric', 'euclidean')
        )
        mu_train = fuzzy_model.fit_transform()
        mu_test = fuzzy_model.transform(dataset.testing_data)

        estimator = EffortEstimator(
            n_features=mu_train.shape[1],
            num_clusters=actual_clusters,
            lr=best_params.get('lr', 1e-3),
            batch_size=16,
            epochs=150,
            weight_decay=best_params.get('weight_decay', 1e-3),
            kernels=best_params.get('kernels', 16),
            target_min=dataset.target_min,
            target_range=dataset.target_range,
            loss_type=best_params.get('loss_type', 'hybrid'),
            alpha=best_params.get('alpha', 0.5),
            use_log_target=dataset.use_log_target
        )
        estimator.fit(
            mu_train,
            dataset.training_target_scaled,
            X_train_raw=dataset.training_data,
            verbose=False,
            patience=20,
            val_ratio=0.15
        )


        mae, rmse, mdae, mmre, mdmre, pred25 = estimator.evaluate(
            mu_test,
            dataset.testing_target_scaled,
            X_test_raw=dataset.testing_data,
            inverse_transform=dataset.inverse_transform_target
        )

        print(f"Outer Fold {fold_idx + 1} Result -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MMRE: {mmre:.4f} | Pred(25): {pred25:.2f}%")
        outer_results.append([mae, rmse, mdae, mmre, mdmre, pred25])

    results_arr = np.array(outer_results)
    means = np.mean(results_arr, axis=0)

    print("\n" + "=" * 70)
    print("FINAL NESTED CROSS-VALIDATION SUMMARY:")
    print(f"Average MAE:     {means[0]:.4f}")
    print(f"Average RMSE:    {means[1]:.4f}")
    print(f"Average MdAE:    {means[2]:.4f}")
    print(f"Average MMRE:    {means[3]:.4f}")
    print(f"Average MdMRE:   {means[4]:.4f}")
    print(f"Average Pred(25):{means[5]:.2f}%")
    print("=" * 70)

    return means

if __name__ == '__main__':
    datasets_config = [
        ('../Software-estimation-datasets/albrecht.arff', 'Effort'),
        ('../Software-estimation-datasets/china.arff', 'Effort'),
        ('../Software-estimation-datasets/maxwell.arff', 'Effort'),
        ('../Software-estimation-datasets/Desharnais.csv', 'Effort'),
        ('../Software-estimation-datasets/COCOMO-81.csv', 'actual')
    ]

    all_dataset_results = {}

    for path, target_col in datasets_config:
        ds_name = os.path.basename(path).split('.')[0].upper()
        print("\n" + "#" * 80)
        print(f"RUNNING NESTED CV PIPELINE FOR DATASET: {ds_name}")
        print("#" * 80)

        results = run_nested_cv(
            dataset_path=path,
            target_column=target_col,
            n_outer_folds=5,
            n_trials=10,
            is_loo=False
        )
        all_dataset_results[ds_name] = results

    print("\n" + "=" * 80)
    print("ALL DATASETS NESTED CROSS-VALIDATION SUMMARY TABLE:")
    print("=" * 80)
    print(f"{'Dataset':<15} | {'MAE':<10} | {'RMSE':<10} | {'MdAE':<10} | {'MMRE':<10} | {'MdMRE':<10} | {'Pred(25)':<10}")
    print("-" * 80)
    for ds_name, res in all_dataset_results.items():
        print(f"{ds_name:<15} | {res[0]:<10.4f} | {res[1]:<10.4f} | {res[2]:<10.4f} | {res[3]:<10.4f} | {res[4]:<10.4f} | {res[5]:<9.2f}%")
    print("=" * 80)

