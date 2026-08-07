import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fuzzy-DCNN-implementation')))

from utils import DATASET_PATHS, run_loocv_experiment, compute_wilcoxon_test

def run_huber_loss_evaluation():
    print("=" * 80)
    print("TEST 4: HUBER LOSS & ROBUST LOSS FUNCTION EVALUATION")
    print("=" * 80)

    output_dir = os.path.dirname(__file__)
    results_csv = os.path.join(output_dir, 'huber_results.csv')

    loss_configs = [
        ('RMSE Loss (Baseline)', 'rmse', 0.01),
        ('MAE Loss', 'mae', 0.01),
        ('Huber Loss (delta=0.01)', 'huber', 0.01),
        ('Huber Loss (delta=0.05)', 'huber', 0.05),
        ('Huber Loss (delta=0.1)', 'huber', 0.1),
        ('Huber Loss (delta=0.2)', 'huber', 0.2),
        ('Huber Loss (delta=0.5)', 'huber', 0.5),
    ]

    seeds = [2501439, 3964827]
    records = []

    for ds_name, ds_path in DATASET_PATHS.items():
        print(f"\n---> Evaluating Loss Functions on dataset: {ds_name.upper()}")

        baseline_preds_by_seed = {}
        baseline_actuals = None

        for seed in seeds:
            b_m, b_act, b_pred = run_loocv_experiment(
                dataset_path=ds_path,
                seed=seed,
                dcnn_epochs=300,
                loss_type='rmse'
            )
            baseline_preds_by_seed[seed] = b_pred
            baseline_actuals = b_act

        for label, ltype, delta in loss_configs:
            maes, medaes, sas, deltas_list = [], [], [], []
            all_preds = []

            for seed in seeds:
                m, actuals, preds = run_loocv_experiment(
                    dataset_path=ds_path,
                    seed=seed,
                    dcnn_epochs=300,
                    loss_type=ltype,
                    huber_delta=delta
                )
                maes.append(m['mae'])
                medaes.append(m['medae'])
                sas.append(m['sa'])
                deltas_list.append(m['delta'])
                all_preds.append(preds)

            mean_preds = np.mean(all_preds, axis=0)
            mean_baseline_preds = np.mean([baseline_preds_by_seed[s] for s in seeds], axis=0)

            wilc = compute_wilcoxon_test(baseline_actuals, mean_baseline_preds, mean_preds)

            record = {
                'dataset': ds_name,
                'config_label': label,
                'loss_type': ltype,
                'huber_delta': delta if ltype == 'huber' else None,
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'medae_mean': float(np.mean(medaes)),
                'medae_std': float(np.std(medaes)),
                'sa_mean': float(np.mean(sas)),
                'delta_effect_mean': float(np.mean(deltas_list)),
                'wilcoxon_p_value': wilc['p_value'],
                'statistically_significant': wilc['statistically_significant'],
                'wins_vs_rmse': wilc['wins'],
                'losses_vs_rmse': wilc['losses'],
                'ties_vs_rmse': wilc['ties']
            }
            records.append(record)

            print(f"  [{label:<24}] MAE: {np.mean(maes):8.2f} ± {np.std(maes):5.2f} | MedAE: {np.mean(medaes):8.2f} | SA: {np.mean(sas):.4f} | p-val: {wilc['p_value']:.4f}")

    df_res = pd.DataFrame(records)
    df_res.to_csv(results_csv, index=False)
    print(f"\nSaved Huber loss evaluation results to {results_csv}")

if __name__ == '__main__':
    run_huber_loss_evaluation()
