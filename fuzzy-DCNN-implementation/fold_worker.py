"""
Logica di un singolo fold LOOCV/K-fold, isolata in un modulo importabile a parte.

Perché questo file esiste separato da evaluate.py:
Su Windows, joblib (backend 'loky') usa processi separati creati con 'spawn', non
'fork'. Se la funzione eseguita nei worker (process_single_fold) fosse definita
direttamente nello script eseguito come __main__ (cioè in evaluate.py quando lanciato
con `python evaluate.py`), cloudpickle dovrebbe serializzarla "per valore" insieme a
tutto lo stato del modulo in cui vive — incluso, se la GPU è stata toccata nel
processo principale (es. torch.cuda.is_available()), il contesto CUDA/cuDNN, che
contiene oggetti nativi non serializzabili (es. 'CudnnModule').
Definendo process_single_fold in un modulo importabile separato, i worker possono
semplicemente IMPORTARE la funzione per riferimento invece di serializzarla per
valore, evitando il crash.
"""

import random
import numpy as np
import torch

from dataset import Dataset
from elbow import elbow_XB
from DCNN import EffortEstimator


def is_china_dataset(dataset_path: str) -> bool:
    """China.arff è l'unico dataset abbastanza grande da giustificare 10-fold CV invece di LOOCV."""
    return 'china' in dataset_path.lower()


def target_column_for(dataset_path: str) -> str:
    """COCOMO usa 'actual' come nome della colonna target; tutti gli altri usano 'Effort'."""
    return 'actual' if 'cocomo' in dataset_path.lower() else 'Effort'


def compute_dynamic_huber_delta(dataset: Dataset, multiplier: float) -> float:
    """
    Calcola huber_delta (in scala target normalizzata [0, 1]) come frazione della
    mediana del target di TRAINING del fold corrente, così la soglia si adatta
    automaticamente alla scala/skewness di ciascun dataset invece di essere un
    valore fisso arbitrario. Usa solo dataset.training_target_unscaled -> nessun
    leakage dal fold di test.
    """
    train_median_original = dataset.training_target_unscaled.median()
    delta_original = multiplier * train_median_original
    delta_normalized = delta_original / dataset.target_range_val
    return float(delta_normalized)


def process_single_fold(dataset_path: str, target_col: str, use_knn_imputation: bool,
                         train_idx, test_idx, random_seed: int, max_fuzzy_clusters: int,
                         dcnn_lr: float, dcnn_epochs: int, dcnn_batch_size: int,
                         loss_type: str, huber_delta_multiplier: float):
    """
    Allena e valuta il modello Fuzzy-DCNN su un singolo fold LOOCV/K-fold.

    Ogni worker fissa i propri seed per garantire determinismo. Con backend='loky'
    (processi separati, non thread) questo è realmente efficace: a differenza del
    threading, non c'è stato RNG globale condiviso tra fold eseguiti in parallelo.
    """
    torch.set_num_threads(1)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # Il GPU viene usato solo per China: per gli altri dataset (piccoli) l'overhead
    # di trasferimento CPU<->GPU rende la CPU più veloce.
    use_gpu = is_china_dataset(dataset_path) and torch.cuda.is_available()
    if use_gpu:
        device = 'cuda'
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = 'cpu'

    # 1. Split + scaling (fit su train, applicato a test) — vedi dataset.py
    dataset = Dataset(
        dataset_path,
        target_column_name=target_col,
        train_indices=train_idx,
        test_indices=test_idx,
        use_knn_imputation=use_knn_imputation,
    )

    # 2. Fuzzificazione: FCM sul training set, scelta di k via indice Xie-Beni
    fuzzy_model, optimal_k, mu_train = elbow_XB(dataset, max_clusters=max_fuzzy_clusters)
    mu_test = fuzzy_model.transform(dataset.testing_data)

    # 3. Training del DCNN sulle membership fuzzy
    target_mean_scaled = dataset.training_target_scaled.mean()
    dynamic_huber_delta = compute_dynamic_huber_delta(dataset, huber_delta_multiplier)
    estimator = EffortEstimator(
        m_features=dataset.training_data.shape[0],
        k_clusters=optimal_k,
        lr=dcnn_lr,
        epochs=dcnn_epochs,
        batch_size=dcnn_batch_size,
        target_min=0.0,
        target_range=1.0,
        target_mean=target_mean_scaled,
        device=device,
        loss_type=loss_type,
        huber_delta=dynamic_huber_delta,
    )
    estimator.fit(mu_train, dataset.training_target_scaled.values)

    # 4. Predizione sul fold di test, riportata alla scala originale
    pred_scaled = estimator.predict(mu_test)
    pred_unscaled = dataset.inverse_transform_target(pred_scaled)

    actuals_fold = dataset.testing_target_unscaled.values
    predictions_fold = pred_unscaled
    return actuals_fold, predictions_fold