import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fuzzy-DCNN-implementation')))

from utils import DATASET_PATHS, run_loocv_experiment, compute_wilcoxon_test

def run_target_transformation_test():
    print("=" * 80)
    print("TEST 5: TARGET LOG-TRANSFORMATION (LOG(1+Y)) EVALUATION")
    print("=" * 80)

    output_dir = os.path.dirname(__file__)
    results_csv = os.path.join(output_dir, 'log_transform_comparison.csv')

    configs = [
        ('Linear Target + RMSE', False, 'rmse'),
        ('Linear Target + Huber(0.01)', False, 'huber'),
        ('Log Target log1p + RMSE', True, 'rmse'),
        ('Log Target log1p + Huber(0.01)', True, 'huber'),
    ]

    seeds = [2501439, 3964827]
    records = []

    for ds_name, ds_path in DATASET_PATHS.items():
        print(f"\n---> Evaluating Target Transformation on dataset: {ds_name.upper()}")

        baseline_preds_by_seed = {}
        baseline_actuals = None

        for seed in seeds:
            b_m, b_act, b_pred = run_loocv_experiment(
                dataset_path=ds_path,
                seed=seed,
                dcnn_epochs=300,
                use_log_target=False,
                loss_type='rmse'
            )
            baseline_preds_by_seed[seed] = b_pred
            baseline_actuals = b_act

        for label, use_log, ltype in configs:
            maes, medaes, sas, deltas_list, dead_rates = [], [], [], [], []
            all_preds = []

            for seed in seeds:
                m, actuals, preds = run_loocv_experiment(
                    dataset_path=ds_path,
                    seed=seed,
                    dcnn_epochs=300,
                    use_log_target=use_log,
                    loss_type=ltype
                )
                maes.append(m['mae'])
                medaes.append(m['medae'])
                sas.append(m['sa'])
                deltas_list.append(m['delta'])
                dead_rates.append(m['dead_neuron_rate'])
                all_preds.append(preds)

            mean_preds = np.mean(all_preds, axis=0)
            mean_baseline_preds = np.mean([baseline_preds_by_seed[s] for s in seeds], axis=0)

            wilc = compute_wilcoxon_test(baseline_actuals, mean_baseline_preds, mean_preds)

            record = {
                'dataset': ds_name,
                'config_label': label,
                'use_log_target': use_log,
                'loss_type': ltype,
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'medae_mean': float(np.mean(medaes)),
                'medae_std': float(np.std(medaes)),
                'sa_mean': float(np.mean(sas)),
                'delta_effect_mean': float(np.mean(deltas_list)),
                'dead_neuron_rate_pct': float(np.mean(dead_rates)),
                'wilcoxon_p_value': wilc['p_value'],
                'statistically_significant': wilc['statistically_significant'],
                'wins_vs_linear_rmse': wilc['wins'],
                'losses_vs_linear_rmse': wilc['losses'],
                'ties_vs_linear_rmse': wilc['ties']
            }
            records.append(record)

            print(f"  [{label:<32}] MAE: {np.mean(maes):8.2f} ± {np.std(maes):5.2f} | MedAE: {np.mean(medaes):8.2f} | SA: {np.mean(sas):.4f} | Dead: {np.mean(dead_rates):5.1f}% | p-val: {wilc['p_value']:.4f}")

    df_res = pd.DataFrame(records)
    df_res.to_csv(results_csv, index=False)
    print(f"\nSaved Target Transformation comparison results to {results_csv}")

if __name__ == '__main__':
    run_target_transformation_test()
