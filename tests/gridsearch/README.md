# Test 3: Ottimizzazione Iperparametrica Efficiente (2-Stage Grid Search)

## Descrizione
Questo modulo esegue la ricerca degli iperparametri ottimali per ciascun dataset, superando le configurazioni di default statiche. Per prevenire l'esplosione combinatoria del tempo di calcolo sotto Leave-One-Out Cross-Validation (LOOCV), si adotta una strategia di **Grid Search a 2 Stadi disaccoppiati**:

1. **Stadio 1 (Architettura Neurale & Regolarizzazione):**
   - Parametri esplorati: `num_filters` $\in [8, 16, 32]$, `dropout_rate` $\in [0.1, 0.2, 0.3]$, `weight_decay` $\in [0.0, 0.001]$.
   - Fissa il learning rate a $10^{-3}$ per isolare la capacità rappresentativa della rete.
2. **Stadio 2 (Ottimizzazione & Dinamica di Addestramento):**
   - Mantiene la migliore configurazione architetturale identificata nello Stadio 1.
   - Parametri esplorati: `dcnn_lr` $\in [0.005, 0.001, 0.0005, 0.0001]$, `dcnn_batch_size` $\in [8, 16, 32]$, `dcnn_epochs` $\in [200, 500, 1000]$ (con early stopping facoltativo).

## Metodologia di Validazione
- LOOCV per *Albrecht*, *COCOMO-81*, *Desharnais*, *Maxwell*.
- 10-Fold Cross-Validation per *China*.
- Calcolo congiunto di MAE, MedAE e Standardized Accuracy (SA).

## Esecuzione
Esecuzione dello script di test dedicato:
```bash
python tests/gridsearch/run_gridsearch_test.py
```

## File di Output
- `grid_results_{dataset}.csv`: Valutazione dettagliata di ogni singolo trial della griglia per ciascun dataset.
- `best_hyperparameters.json`: File JSON riassuntivo contenente i parametri ottimali vincenti consolidati per ogni dataset.
- `grid_results_{dataset}_fab.csv` & `best_hyperparameters_fab.json`: Varianti sperimentali di confronto indipendente.
