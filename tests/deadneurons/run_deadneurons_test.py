import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fuzzy-DCNN-implementation')))

from utils import DATASET_PATHS, resolve_dataset_path, run_loocv_experiment, compute_wilcoxon_test

def run_dead_neuron_analysis():
    print("=" * 80)
    print("TEST 2: DEAD NEURON ANALYSIS & ACTIVATION FUNCTION COMPARISON")
    print("=" * 80)

    output_dir = os.path.dirname(__file__)
    results_csv = os.path.join(output_dir, 'activation_comparison.csv')

    variants = [
        ('Variant A (ReLU)', 'relu'),
        ('Variant B (Softplus)', 'softplus'),
        ('Variant C (Linear + Clip - Baseline)', 'linear_clip'),
        ('Variant D (LeakyReLU + Clip)', 'leaky_relu'),
        ('Variant E (ELU + 1)', 'elu'),
    ]

    seeds = [42, 123, 456]
    records = []

    # Focus on China and Desharnais, plus all datasets
    target_datasets = ['china', 'desharnais', 'cocomo', 'albrecht', 'maxwell']

    for ds_name in target_datasets:
        if ds_name not in DATASET_PATHS:
            continue

        print(f"\n---> Analyzing activation variants on dataset: {ds_name.upper()}")
        ds_path = DATASET_PATHS[ds_name]

        # First run baseline (Variant C) across seeds to collect baseline predictions
        baseline_preds_by_seed = {}
        baseline_actuals = None

        for seed in seeds:
            b_m, b_act, b_pred = run_loocv_experiment(
                dataset_path=ds_path,
                seed=seed,
                dcnn_epochs=300,
                activation_type='linear_clip'
            )
            baseline_preds_by_seed[seed] = b_pred
            baseline_actuals = b_act

        for var_label, act_type in variants:
            maes, medaes, sas, deltas, dead_rates = [], [], [], [], []
            all_preds = []

            for seed in seeds:
                m, actuals, preds = run_loocv_experiment(
                    dataset_path=ds_path,
                    seed=seed,
                    dcnn_epochs=300,
                    activation_type=act_type
                )
                maes.append(m['mae'])
                medaes.append(m['medae'])
                sas.append(m['sa'])
                deltas.append(m['delta'])
                dead_rates.append(m['dead_neuron_rate'])
                all_preds.append(preds)

            mean_preds = np.mean(all_preds, axis=0)
            mean_baseline_preds = np.mean([baseline_preds_by_seed[s] for s in seeds], axis=0)

            wilc = compute_wilcoxon_test(baseline_actuals, mean_baseline_preds, mean_preds)

            record = {
                'dataset': ds_name,
                'variant': var_label,
                'activation_type': act_type,
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'medae_mean': float(np.mean(medaes)),
                'medae_std': float(np.std(medaes)),
                'sa_mean': float(np.mean(sas)),
                'delta_mean': float(np.mean(deltas)),
                'dead_neuron_rate_pct': float(np.mean(dead_rates)),
                'wilcoxon_p_value': wilc['p_value'],
                'statistically_significant': wilc['statistically_significant'],
                'wins_vs_baseline': wilc['wins'],
                'losses_vs_baseline': wilc['losses'],
                'ties_vs_baseline': wilc['ties']
            }
            records.append(record)

            print(f"  [{act_type:<12}] Dead Neurons: {np.mean(dead_rates):5.1f}% | MAE: {np.mean(maes):8.2f} ± {np.std(maes):5.2f} | MedAE: {np.mean(medaes):8.2f} | SA: {np.mean(sas):.4f} | p-val: {wilc['p_value']:.4f}")

    df_res = pd.DataFrame(records)
    df_res.to_csv(results_csv, index=False)
    print(f"\nSaved activation comparison results to {results_csv}")

if __name__ == '__main__':
    run_dead_neuron_analysis()
