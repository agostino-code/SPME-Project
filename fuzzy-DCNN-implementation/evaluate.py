import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut, KFold
from joblib import Parallel, delayed

from dataset import Dataset
from fold_worker import is_china_dataset, target_column_for, process_single_fold

sys.stdout.reconfigure(line_buffering=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
CV_SPLIT_SEED = 42
N_RANDOM_BASELINE_RUNS = 1000   # numero di permutazioni casuali usate per stimare SA / Delta
MAX_FUZZY_CLUSTERS = 10         # limite superiore per la ricerca del k ottimale (indice Xie-Beni)

DCNN_LEARNING_RATE = 0.001
DCNN_EPOCHS = 1000
DCNN_BATCH_SIZE = 16

# Loss function: 'rmse' (default originale, sensibile agli outlier) o 'huber'
# (quadratica per errori piccoli, lineare per errori grandi -> meno dominata dagli
# outlier).
DCNN_LOSS_TYPE = 'huber'

CHINA_KFOLD_SPLITS = 10         # China usa 10-fold invece di LOOCV (dataset troppo grande per LOOCV)

DATASETS_TO_EVALUATE = [
    '../Software-estimation-datasets/COCOMO-81.csv',
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(actual, predicted):
    """
    Calcola MAE, MedAE, Standardized Accuracy (SA) ed effect size (Delta), confrontando
    l'errore del modello con quello di N_RANDOM_BASELINE_RUNS predittori casuali
    (permutazioni delle predizioni), come descritto nel paper (Eq. 8-10).

    MedAE (mediana degli errori assoluti) viene riportata perché, su dataset piccoli
    e fortemente skewed come COCOMO-81, pochi fold con outlier estremi dominano il MAE
    aggregato mentre la MedAE riflette l'errore "tipico" su un progetto medio — utile
    per capire se il modello generalizza bene sulla maggioranza dei casi anche quando
    il MAE resta alto a causa di 2-3 fold catastrofici.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    abs_errors = np.abs(actual - predicted)
    mae = np.mean(abs_errors)
    medae = np.median(abs_errors)

    random_maes = np.zeros(N_RANDOM_BASELINE_RUNS)
    for i in range(N_RANDOM_BASELINE_RUNS):
        np.random.seed(i)
        pred_random = np.random.permutation(actual)
        random_maes[i] = np.mean(np.abs(actual - pred_random))

    mae_random = np.mean(random_maes)
    sd_random = np.std(random_maes)

    sa = 1.0 - (mae / mae_random) if mae_random != 0 else 0.0
    delta = (mae_random - mae) / sd_random if sd_random != 0 else 0.0

    return mae, medae, sa, delta


# --------------------------------------------------------------------------- #
# Full LOOCV/K-fold run for one dataset
# --------------------------------------------------------------------------- #

def run_loocv(dataset_path: str, use_knn_imputation: bool = False,
              seed: int = None, loss_type: str = None,
              n_jobs: int = None, cluster_method: str | int = 'XB-index'):
    """
    Esegue la validazione completa (LOOCV, o 10-fold per China) su un dataset,
    salva i risultati per-fold in CSV e restituisce (MAE, MedAE, SA, Delta) aggregati.
    """
    model_seed = seed if seed is not None else RANDOM_SEED
    loss_type = loss_type if loss_type is not None else DCNN_LOSS_TYPE

    print(f"\n--- Running LOOCV for {os.path.basename(dataset_path)} "
          f"(KNN={use_knn_imputation}) ---", flush=True)

    target_col = target_column_for(dataset_path)
    ds_full = Dataset(dataset_path, target_column_name=target_col,
                       use_knn_imputation=use_knn_imputation, verbose=True)

    if is_china_dataset(dataset_path):
        cv_seed = CV_SPLIT_SEED if 'CV_SPLIT_SEED' in globals() else 42
        cv = KFold(n_splits=CHINA_KFOLD_SPLITS, shuffle=True, random_state=cv_seed)
        print("NOTE: Using 10-fold CV (not LOOCV) for China as a preliminary check.", flush=True)
    else:
        cv = LeaveOneOut()

    # Un job per GPU (evita context-switching/VRAM OOM), tutti i core per CPU.
    use_gpu = is_china_dataset(dataset_path) and torch.cuda.is_available()
    num_jobs = n_jobs if n_jobs is not None else (1 if use_gpu else -1)

    # backend='loky' esplicito: ogni fold gira in un processo separato, con memoria e
    # stato RNG (numpy/torch) completamente isolati. Con 'threading' (a cui joblib può
    # fare fallback silenzioso) i fold condividerebbero lo stesso stato RNG globale,
    # rendendo i seed fissati in process_single_fold inefficaci in caso di esecuzione
    # concorrente -> risultati non riproducibili run-to-run nonostante i seed.
    # Richiede che process_single_fold sia importata da un modulo separato (fold_worker.py),
    # non definita in questo script __main__ (vedi nota in testa al file).
    fold_results = Parallel(n_jobs=num_jobs, backend='loky', verbose=5)(
        delayed(process_single_fold)(
            dataset_path, target_col, use_knn_imputation, train_idx, test_idx,
            random_seed=model_seed,
            max_fuzzy_clusters=MAX_FUZZY_CLUSTERS,
            dcnn_lr=DCNN_LEARNING_RATE,
            dcnn_epochs=DCNN_EPOCHS,
            dcnn_batch_size=DCNN_BATCH_SIZE,
            loss_type=loss_type,
            cluster_method=cluster_method
        )
        for train_idx, test_idx in cv.split(ds_full.data)
    )

    # --- Aggregazione dei risultati di tutti i fold ---
    actuals, predictions, fold_indices = [], [], []
    for fold_idx, (fold_actuals, fold_predictions) in enumerate(fold_results):
        actuals.extend(fold_actuals)
        predictions.extend(fold_predictions)
        fold_indices.extend([fold_idx] * len(fold_actuals))

    mae, medae, sa, delta = compute_metrics(actuals, predictions)

    # --- Salvataggio dei risultati per-fold (utile per diagnosi outlier/varianza) ---
    df_results = pd.DataFrame({
        'fold_index': fold_indices,
        'actual': actuals,
        'predicted': predictions,
    })
    df_results['abs_error'] = np.abs(df_results['actual'] - df_results['predicted'])
    df_results['squared_error'] = df_results['abs_error'] ** 2

    ds_name_no_ext = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Costruisci un identificatore per la configurazione
    config_parts = ['FDCNN', loss_type]
    if cluster_method == 'fixed':
        config_parts.append(f"k{cluster_method}")
    config_parts.append(f"seed{model_seed}")
    
    config_str = "_".join(config_parts)
    csv_filename = f"results_per_fold_{ds_name_no_ext}__{config_str}.csv"
    
    df_results.to_csv(csv_filename, index=False)
    print(f"Saved per-fold results to {csv_filename}")

    print(f"Results for {os.path.basename(dataset_path)}:")
    print(f"MAE:   {mae:.4f}")
    print(f"MedAE: {medae:.4f}")
    print(f"SA:    {sa:.4f}")
    print(f"Delta: {delta:.4f}")

    return mae, medae, sa, delta, csv_filename


# --------------------------------------------------------------------------- #
# Entry point: run all configured datasets and print a summary table
# --------------------------------------------------------------------------- #

def main():
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print("Note: Using GPU ONLY for China.arff. CPU is faster for other small "
              "datasets due to transfer overhead.")
    else:
        print("No GPU detected.")

    results = {}
    for ds_path in DATASETS_TO_EVALUATE:
        if not os.path.exists(ds_path):
            print(f"File not found: {ds_path}")
            continue

        start_time = time.time()
        mae, medae, sa, delta, _ = run_loocv(ds_path)
        elapsed = time.time() - start_time

        method = '10-fold' if is_china_dataset(ds_path) else 'LOOCV'
        results[os.path.basename(ds_path)] = {
            'MAE': mae, 'MedAE': medae, 'SA': sa, 'Delta': delta, 'Time': elapsed, 'Method': method,
        }
        print(f"Time for {os.path.basename(ds_path)}: {elapsed:.2f} seconds")

    print_summary_table(results)


def print_summary_table(results: dict):
    print("\n" + "=" * 90)
    print("FINAL SUMMARY (POST-FIX LOOCV / 10-FOLD)")
    print("=" * 90)
    print(f"{'Dataset':<15} | {'Method':<10} | {'MAE':<10} | {'MedAE':<10} | {'SA':<10} | "
          f"{'Delta':<10} | {'Time (s)':<10}")
    for name, r in results.items():
        print(f"{name:<15} | {r['Method']:<10} | {r['MAE']:<10.2f} | {r['MedAE']:<10.2f} | "
              f"{r['SA']:<10.2f} | {r['Delta']:<10.2f} | {r['Time']:<10.2f}")
    print("=" * 90)


if __name__ == '__main__':
    main()