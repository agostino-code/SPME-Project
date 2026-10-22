# Test Suite: Baseline Models Comparison

## Descrizione
Questo modulo esegue la valutazione comparativa rigorosa dei modelli baseline classici della letteratura di *Software Development Effort Estimation* (SDEE):
1. **Metodo Di Martino et al. (2020):** Simple Linear Regression (SLR) applicata alla metrica di dimensione funzionale primaria (o linee di codice).
2. **Random Forest Regressor (RF):** Modello ensemble non lineare basato su alberi decisionali.
3. **Support Vector Regression (SVR / SVM):** Modello basato su iperpiani di regressione e kernel RBF su feature standardizzate.
4. **Nostro Modello Fuzzy-DCNN Ottimizzato:** Rete convoluzionale fuzzy con Huber Loss ($\delta=0.01$), target logaritmico e output Linear+Clip.

## Protocollo di Validazione
- **LOOCV (Leave-One-Out Cross-Validation):** Eseguito sui 4 dataset a campionamento ridotto (*COCOMO-81*, *Albrecht*, *Desharnais*, *Maxwell*).
- **10-Fold Cross-Validation:** Eseguito sul dataset esteso *China* ($N=499$) con seed di partizione controllato (`seed=42`).

## Esecuzione
```bash
python tests/baselines/run_baselines_test.py
```

## Risultati di Confronto (MedAE e Standardized Accuracy)

| Dataset | Di Martino (SLR) MedAE | Random Forest MedAE | SVM (SVR) MedAE | **Nostro Fuzzy-DCNN MedAE** | **Nostro SA (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **China** | 1757.98 h | 528.10 h | 226.88 h | **117.09 h** | **83.0%** |
| **COCOMO-81** | 185.87 m | 196.19 m | 161.99 m | **71.24 m** | **51.0%** |
| **Albrecht** | 7.58 m | 4.29 m | 4.13 m | **3.11 m** | **76.0%** |
| **Desharnais** | 1267.56 h | 1375.61 h | 1094.91 h | **995.24 h** | **50.0%** |
| **Maxwell** | 2503.58 h | 2099.73 h | 1732.94 h | **1986.93 h** | **56.0%** |

### File di Output
- `baseline_comparison_results.csv`: Risultati dettagliati per modello, dataset, MAE, MedAE e SA.
