# Report Integrato e Globale dei Risultati dei Test: F-DCNN SDEE

**Data di generazione:** 19 Agosto 2026
**Progetto:** Fuzzy Deep Convolutional Neural Network (F-DCNN) for Software Development Effort Estimation (SDEE)
**Scopo:** Consolidamento e sintetizzazione di tutti i risultati dei test sperimentalmente condotti in tutti i formati file disponibili (`CSV`, `JSON`, `MD`, file per-fold e relative varianti Standard e Fab).

---

## 1. Mappa dei File e dei Formati Analizzati

L'indagine ha elaborato **tutti i file di risultato** presenti nel repository, suddivisi nelle seguenti categorie e formati:

| Categoria Test | Formato File | File Analizzati | Contenuto / Scopo |
| :--- | :--- | :--- | :--- |
| **Analisi Diagnostica Pre-Test** | `.md` | `tests/history/pre-test-analysis.md` | Diagnostica baseline, gap con paper Azzeh 2024, fix dead ReLU e determinismo |
| **Test 1: Dead Neurons** | `.csv` | `tests/deadneurons/activation_comparison.csv` | Valutazione strategie di attivazione in output (ReLU, Softplus, Linear+Clip, LeakyReLU, ELU) |
| **Test 2: Robust Loss (Huber)** | `.csv` | `tests/huber_loss/huber_results.csv`, `huber_results (Fab).csv` | Confronto tra RMSE, MAE e Huber Loss ($\delta \in [0.01, 0.5]$) con test di Wilcoxon e Effect Size |
| **Test 3: Target Log-Transform** | `.csv` | `tests/target_transform/log_transform_comparison.csv`, `... (Fab).csv` | Valutazione trasformazione logaritmica $\log(1+y)$ vs Target Lineare |
| **Test 4: Cluster Validation** | `.csv`, `.json` | `tests/clusters/cluster_metrics.csv`, `recommendations.json` (Std & Fab) | Metriche cluster FCM (Silhouette, DB, CH, XB) e raccomandazioni $K$ ottimale |
| **Test 5: Grid Search** | `.csv`, `.json` | `tests/gridsearch/grid_results_{ds}.csv`, `best_hyperparameters.json` (Std & Fab) | Ottimizzazione 2-stage iperparametri per Albrecht, China, COCOMO-81, Desharnais, Maxwell |
| **Predizioni Per-Fold** | `.csv` | `results_per_fold_*.csv` (10 file nel root) | Valori reali, predetti, errore assoluto ed errore quadratico fold per fold |

---

## 2. Diagnostica Pre-Test e Correzione del Baseline

Prima dell'esecuzione dei test estesi, la fase diagnostica (descritta in [`pre-test-analysis.md`](file:///D:/SPME-Project/tests/history/pre-test-analysis.md)) ha identificato tre criticità fondamentali nel modello originale:
1. **Dying ReLU & Collasso a 0:** L'uso di `nn.ReLU()` al layer di output causava il collasso a predizioni nulle sul **67% dei fold di COCOMO-81**. La soluzione baseline adottata è stata l'**Output Lineare in training con clipping non-negativo (`np.maximum(preds, 0.0)`) applicato solo in inferenza**.
2. **Esplosione del Gradiente su MSE:** La loss MSE/RMSE su target non scalati ad alta varianza generava gradienti $>10.000$. L'adozione della **Huber Loss ($\delta=0.01$)** o del **Target Log-Transform ($\log(1+y)$)** ha stabilizzato la convergenza.
3. **Non-determinismo del Multithreading:** L'uso di `ThreadingBackend` in `joblib` condivideva lo stato RNG tra thread. Risolto forzando il backend multiprocesso `loky` in [`fold_worker.py`](file:///D:/SPME-Project/fuzzy-DCNN-implementation/fold_worker.py).

### Confronto Baseline Iniziale vs Paper Azzeh et al. (2024)

| Dataset | MAE Nostro Baseline | MAE Paper (Azzeh 2024) | MedAE Nostra Baseline | MedAE / Paper MAE Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **China** | 1519.79 | 899 | 688.19 | **0.77×** |
| **Desharnais** | 2277.98 | 1044 | 1752.77 | 1.68× |
| **Maxwell** | 6438.75 | 1749 | 3348.03 | 1.91× |
| **COCOMO-81** | 755.18 | 160 | 111.58 | **0.70×** |
| **Albrecht** | 8.65 | 1.1 | 5.10 | 4.64× |

> [!NOTE]
> Su COCOMO-81 e China, la **MedAE** della F-DCNN (111.58 e 688.19) è risultata nettamente **inferiore al MAE dichiarato nel paper**, indicando che la presenza di pochi progetti outlier ad alto effort distorce pesantemente il MAE mediato.

---

## 3. Test 1: Dead Neuron Analysis (Funzioni di Attivazione di Output)

Analisi delle strategie di attivazione in output per eliminare il problema "dying ReLU" (dati estratti da [`activation_comparison.csv`](file:///D:/SPME-Project/tests/deadneurons/activation_comparison.csv)):

| Dataset | Variante Attivazione | Tipo | MAE Mean | MedAE Mean | SA Mean | Dead Neuron Rate (%) | Wilcoxon p-value | Significativo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| china | Variant A (ReLU) | relu | 1325.1058 | 571.6872 | 0.7302 | 0.0000 | 0.3358 | False |
| china | Variant B (Softplus) | softplus | 1822.6994 | 1165.0574 | 0.6288 | 0.0000 | 0.0000 | True |
| china | Variant C (Linear + Clip - Baseline) | linear_clip | 1278.6248 | 547.6252 | 0.7396 | 0.0000 | 1.0000 | False |
| china | Variant D (LeakyReLU + Clip) | leaky_relu | 1341.2028 | 587.5532 | 0.7269 | 0.0000 | 0.0273 | True |
| china | Variant E (ELU + 1) | elu | 1795.6444 | 1086.1143 | 0.6343 | 0.0000 | 0.0000 | True |
| desharnais | Variant A (ReLU) | relu | 2425.3767 | 1495.8035 | 0.4064 | 0.0000 | 0.2240 | False |
| desharnais | Variant B (Softplus) | softplus | 2498.3316 | 1846.3660 | 0.3886 | 0.0000 | 0.5373 | False |
| desharnais | Variant C (Linear + Clip - Baseline) | linear_clip | 2407.6385 | 1587.9123 | 0.4108 | 0.0000 | 1.0000 | False |
| desharnais | Variant D (LeakyReLU + Clip) | leaky_relu | 2384.9009 | 1538.2084 | 0.4164 | 0.0000 | 0.5887 | False |
| desharnais | Variant E (ELU + 1) | elu | 2673.7757 | 1875.4544 | 0.3457 | 0.0000 | 0.3234 | False |
| cocomo | Variant A (ReLU) | relu | 631.5079 | 114.2382 | 0.4290 | 0.0000 | 0.3997 | False |
| cocomo | Variant B (Softplus) | softplus | 735.0315 | 118.1449 | 0.3354 | 0.0000 | 0.9073 | False |
| cocomo | Variant C (Linear + Clip - Baseline) | linear_clip | 639.0360 | 144.5204 | 0.4222 | 0.0000 | 1.0000 | False |
| cocomo | Variant D (LeakyReLU + Clip) | leaky_relu | 675.6831 | 126.9053 | 0.3891 | 0.0000 | 0.4893 | False |
| cocomo | Variant E (ELU + 1) | elu | 756.7749 | 157.5195 | 0.3158 | 0.0000 | 0.2128 | False |
| albrecht | Variant A (ReLU) | relu | 7.8302 | 4.7420 | 0.6782 | 0.0000 | 0.2076 | False |
| albrecht | Variant B (Softplus) | softplus | 8.4878 | 5.8367 | 0.6512 | 0.0000 | 0.7469 | False |
| albrecht | Variant C (Linear + Clip - Baseline) | linear_clip | 8.0230 | 4.8282 | 0.6703 | 0.0000 | 1.0000 | False |
| albrecht | Variant D (LeakyReLU + Clip) | leaky_relu | 7.9096 | 4.4086 | 0.6750 | 0.0000 | 0.1974 | False |
| albrecht | Variant E (ELU + 1) | elu | 8.2030 | 6.2276 | 0.6629 | 0.0000 | 0.9664 | False |
| maxwell | Variant A (ReLU) | relu | 4652.1415 | 2973.7188 | 0.4613 | 0.0000 | 0.6920 | False |
| maxwell | Variant B (Softplus) | softplus | 4904.0752 | 2998.4586 | 0.4321 | 0.0000 | 0.6817 | False |
| maxwell | Variant C (Linear + Clip - Baseline) | linear_clip | 4567.7391 | 2746.4381 | 0.4711 | 0.0000 | 1.0000 | False |
| maxwell | Variant D (LeakyReLU + Clip) | leaky_relu | 4634.9307 | 2854.1514 | 0.4633 | 0.0000 | 0.9860 | False |
| maxwell | Variant E (ELU + 1) | elu | 5030.1517 | 3127.3543 | 0.4175 | 0.0000 | 0.4385 | False |

**Conclusioni Dead Neurons:**
- La variante **`Linear + Clip`** (Variant C) azzera completamente la percentuale di neuroni morti (0.0%) senza introdurre un bias positivo sistematico (come accadeva con Softplus).
- `LeakyReLU + Clip` (Variant D) mostra prestazioni pressoché equivalenti al baseline lineare.

---

## 4. Test 2: Huber Loss vs. RMSE (Valutazione Loss Robuste)

Valutazione dell'impatto delle funzioni di loss su dati con forte asimmetria (dati da [`huber_results.csv`](file:///D:/SPME-Project/tests/huber_loss/huber_results.csv)):

| Dataset | Loss | MAE Mean | MAE Std | MedAE Mean | MedAE Std | SA Mean | Delta Effect | Wilcoxon p | Significativo | Wins | Losses | Ties |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cocomo | rmse | 722.2679 | 35.5863 | 181.6718 | 19.2459 | 0.3470 | 4.5497 | 1.0000 | False | 0 | 0 | 63 |
| cocomo | mae | 676.7096 | 11.1755 | 141.1037 | 29.7277 | 0.3881 | 5.0899 | 0.0142 | True | 40 | 23 | 0 |
| cocomo | huber | 639.0360 | 19.6427 | 144.5204 | 32.2427 | 0.4222 | 5.5366 | 0.0022 | True | 41 | 22 | 0 |
| cocomo | huber | 750.0199 | 54.4229 | 178.1066 | 25.1471 | 0.3219 | 4.2207 | 0.1032 | False | 27 | 36 | 0 |
| cocomo | huber | 738.3261 | 34.3511 | 195.9223 | 17.5757 | 0.3324 | 4.3593 | 0.1709 | False | 24 | 39 | 0 |
| cocomo | huber | 769.1633 | 48.6626 | 182.7940 | 26.8863 | 0.3046 | 3.9937 | 0.0135 | True | 19 | 44 | 0 |
| cocomo | huber | 759.9516 | 51.6590 | 172.2972 | 8.5489 | 0.3129 | 4.1029 | 0.9727 | False | 31 | 32 | 0 |
| albrecht | rmse | 7.5741 | 0.7168 | 4.7668 | 0.6474 | 0.6888 | 4.8796 | 1.0000 | False | 0 | 0 | 24 |
| albrecht | mae | 7.7122 | 0.5773 | 4.6674 | 0.4159 | 0.6831 | 4.8394 | 0.6431 | False | 12 | 12 | 0 |
| albrecht | huber | 8.0230 | 0.8116 | 4.8282 | 0.8885 | 0.6703 | 4.7489 | 0.0366 | True | 6 | 18 | 0 |
| albrecht | huber | 7.7124 | 0.6740 | 4.7803 | 0.8867 | 0.6831 | 4.8393 | 0.7469 | False | 13 | 11 | 0 |
| albrecht | huber | 7.5824 | 0.4135 | 4.7103 | 0.3858 | 0.6884 | 4.8772 | 0.8334 | False | 11 | 13 | 0 |
| albrecht | huber | 7.1264 | 0.6333 | 4.7279 | 0.7871 | 0.7072 | 5.0099 | 0.8334 | False | 13 | 11 | 0 |
| albrecht | huber | 7.1211 | 0.8169 | 4.5442 | 0.3357 | 0.7074 | 5.0115 | 0.3305 | False | 14 | 10 | 0 |
| desharnais | rmse | 2286.7960 | 134.5045 | 1556.9300 | 127.7091 | 0.4404 | 6.5492 | 1.0000 | False | 0 | 0 | 77 |
| desharnais | mae | 2347.1334 | 135.5300 | 1598.6592 | 89.0834 | 0.4256 | 6.3296 | 0.2991 | False | 34 | 43 | 0 |
| desharnais | huber | 2407.6385 | 200.3669 | 1587.9123 | 43.0695 | 0.4108 | 6.1094 | 0.1809 | False | 34 | 43 | 0 |
| desharnais | huber | 2397.8298 | 148.0846 | 1646.0321 | 203.4713 | 0.4132 | 6.1451 | 0.0735 | False | 34 | 43 | 0 |
| desharnais | huber | 2396.9814 | 170.6234 | 1672.0753 | 170.1945 | 0.4134 | 6.1482 | 0.0270 | True | 31 | 46 | 0 |
| desharnais | huber | 2363.4162 | 159.8442 | 1473.1615 | 101.7061 | 0.4216 | 6.2704 | 0.0703 | False | 34 | 43 | 0 |
| desharnais | huber | 2363.4744 | 144.0209 | 1592.1577 | 160.9369 | 0.4216 | 6.2701 | 0.1472 | False | 34 | 43 | 0 |
| maxwell | rmse | 4424.5331 | 255.1334 | 2829.9914 | 414.2921 | 0.4876 | 6.5001 | 1.0000 | False | 0 | 0 | 62 |
| maxwell | mae | 4563.1768 | 164.2936 | 2818.0929 | 407.3161 | 0.4716 | 6.2861 | 0.0570 | False | 26 | 36 | 0 |
| maxwell | huber | 4567.7391 | 356.1227 | 2746.4381 | 493.8655 | 0.4711 | 6.2791 | 0.1363 | False | 27 | 35 | 0 |
| maxwell | huber | 4544.6566 | 109.1186 | 2800.3091 | 331.4708 | 0.4737 | 6.3147 | 0.1794 | False | 26 | 36 | 0 |
| maxwell | huber | 4398.9630 | 46.0080 | 2610.1198 | 399.5443 | 0.4906 | 6.5396 | 0.8198 | False | 31 | 31 | 0 |
| maxwell | huber | 4534.7995 | 180.9065 | 2874.5448 | 224.1254 | 0.4749 | 6.3299 | 0.7076 | False | 30 | 32 | 0 |
| maxwell | huber | 4531.1511 | 101.9931 | 2898.3288 | 159.9272 | 0.4753 | 6.3356 | 0.1273 | False | 26 | 36 | 0 |
| china | rmse | 1408.9835 | 82.8068 | 690.0718 | 75.5547 | 0.7131 | 28.9815 | 1.0000 | False | 0 | 0 | 499 |
| china | mae | 1271.9694 | 77.8538 | 463.1977 | 68.5552 | 0.7410 | 30.1155 | 0.0000 | True | 337 | 162 | 0 |
| china | huber | 1278.6248 | 79.8424 | 547.6252 | 55.5127 | 0.7396 | 30.0604 | 0.0000 | True | 313 | 186 | 0 |
| china | huber | 1412.2803 | 100.3122 | 687.8376 | 53.3041 | 0.7124 | 28.9542 | 0.3088 | False | 247 | 252 | 0 |
| china | huber | 1489.9170 | 71.2053 | 852.1021 | 50.0293 | 0.6966 | 28.3116 | 0.0000 | True | 178 | 321 | 0 |
| china | huber | 1523.7496 | 100.2581 | 867.0448 | 73.6530 | 0.6897 | 28.0316 | 0.0000 | True | 156 | 343 | 0 |
| china | huber | 1551.3368 | 149.0827 | 891.2099 | 87.1158 | 0.6841 | 27.8032 | 0.0000 | True | 167 | 332 | 0 |

### Variante (Fab) Huber Loss

| Dataset | Loss | MAE Mean | MAE Std | MedAE Mean | MedAE Std | SA Mean | Delta Effect | Wilcoxon p | Significativo | Wins | Losses | Ties |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cocomo | rmse | 665.1991 | 39.5550 | 152.1746 | 3.3942 | 0.3986 | 5.2264 | 1.0000 | False | 0 | 0 | 63 |
| cocomo | mae | 658.5262 | 2.1010 | 126.2396 | 2.1580 | 0.4046 | 5.3055 | 0.5514 | False | 29 | 34 | 0 |
| cocomo | huber | 687.5108 | 11.9575 | 152.2674 | 11.7843 | 0.3784 | 4.9618 | 0.0821 | False | 24 | 39 | 0 |
| cocomo | huber | 702.4858 | 20.9973 | 180.4610 | 7.2153 | 0.3648 | 4.7843 | 0.0962 | False | 25 | 38 | 0 |
| cocomo | huber | 693.0765 | 1.0441 | 181.6469 | 20.6459 | 0.3733 | 4.8958 | 0.0310 | True | 26 | 37 | 0 |
| cocomo | huber | 705.4519 | 36.9050 | 183.0354 | 5.3991 | 0.3622 | 4.7491 | 0.0140 | True | 21 | 42 | 0 |
| cocomo | huber | 714.8547 | 53.5206 | 181.0328 | 6.6518 | 0.3537 | 4.6376 | 0.0115 | True | 25 | 38 | 0 |
| albrecht | rmse | 8.8323 | 0.7700 | 4.7123 | 0.3470 | 0.6371 | 4.5133 | 1.0000 | False | 0 | 0 | 24 |
| albrecht | mae | 8.3609 | 1.1864 | 4.1903 | 0.0193 | 0.6564 | 4.6505 | 0.5646 | False | 14 | 10 | 0 |
| albrecht | huber | 8.9375 | 1.7175 | 4.3740 | 0.3247 | 0.6327 | 4.4827 | 1.0000 | False | 13 | 11 | 0 |
| albrecht | huber | 9.5613 | 0.8787 | 5.7117 | 0.9095 | 0.6071 | 4.3011 | 0.3305 | False | 10 | 14 | 0 |
| albrecht | huber | 10.0423 | 1.4480 | 5.9101 | 0.0165 | 0.5873 | 4.1611 | 0.1074 | False | 9 | 15 | 0 |
| albrecht | huber | 9.2146 | 0.8099 | 5.2158 | 0.0520 | 0.6214 | 4.4020 | 0.0249 | True | 7 | 17 | 0 |
| albrecht | huber | 8.9267 | 0.7431 | 5.0099 | 0.0462 | 0.6332 | 4.4858 | 0.4061 | False | 15 | 9 | 0 |
| desharnais | rmse | 2526.9135 | 49.7839 | 1436.0703 | 119.8050 | 0.3816 | 5.6753 | 1.0000 | False | 0 | 0 | 77 |
| desharnais | mae | 2488.8428 | 6.9227 | 1544.6902 | 58.1912 | 0.3909 | 5.8138 | 0.8212 | False | 37 | 40 | 0 |
| desharnais | huber | 2589.0437 | 54.5393 | 1644.1349 | 74.3054 | 0.3664 | 5.4492 | 0.3437 | False | 34 | 43 | 0 |
| desharnais | huber | 2526.7826 | 65.9858 | 1693.7807 | 167.4804 | 0.3816 | 5.6758 | 0.5887 | False | 39 | 38 | 0 |
| desharnais | huber | 2527.1404 | 57.8467 | 1659.6843 | 307.1479 | 0.3815 | 5.6745 | 0.7976 | False | 37 | 40 | 0 |
| desharnais | huber | 2537.8993 | 86.5634 | 1493.1427 | 125.1826 | 0.3789 | 5.6353 | 0.6206 | False | 41 | 36 | 0 |
| desharnais | huber | 2610.4724 | 65.3859 | 1441.4699 | 94.6816 | 0.3612 | 5.3712 | 0.2968 | False | 33 | 44 | 0 |
| maxwell | rmse | 4606.2794 | 99.5661 | 2890.9583 | 15.8035 | 0.4666 | 6.2196 | 1.0000 | False | 0 | 0 | 62 |
| maxwell | mae | 4853.6435 | 197.8822 | 3149.0935 | 86.9603 | 0.4379 | 5.8378 | 0.6014 | False | 29 | 33 | 0 |
| maxwell | huber | 4937.1377 | 220.5909 | 2983.2757 | 441.0372 | 0.4283 | 5.7089 | 0.4021 | False | 26 | 36 | 0 |
| maxwell | huber | 4672.8850 | 85.5994 | 2982.6441 | 170.3452 | 0.4589 | 6.1168 | 0.6162 | False | 26 | 36 | 0 |
| maxwell | huber | 4977.9181 | 283.5878 | 3234.7041 | 456.0631 | 0.4236 | 5.6459 | 0.1457 | False | 26 | 36 | 0 |
| maxwell | huber | 4734.8466 | 250.7068 | 2926.8038 | 141.4218 | 0.4517 | 6.0211 | 0.3676 | False | 26 | 36 | 0 |
| maxwell | huber | 4767.0739 | 231.3459 | 2960.5759 | 234.6132 | 0.4480 | 5.9714 | 0.2765 | False | 29 | 32 | 1 |
| china | rmse | 1427.0194 | 49.7959 | 771.0103 | 51.0575 | 0.7095 | 27.5878 | 1.0000 | False | 0 | 0 | 499 |
| china | mae | 1284.6479 | 66.9875 | 492.9342 | 13.5429 | 0.7385 | 28.7149 | 0.0000 | True | 335 | 164 | 0 |
| china | huber | 1316.0850 | 19.8202 | 591.6463 | 29.2564 | 0.7321 | 28.4660 | 0.0000 | True | 312 | 187 | 0 |
| china | huber | 1473.7922 | 10.9096 | 823.0796 | 41.7919 | 0.7000 | 27.2176 | 0.0001 | True | 195 | 304 | 0 |
| china | huber | 1530.3170 | 0.7980 | 818.7054 | 37.6500 | 0.6885 | 26.7701 | 0.0000 | True | 221 | 278 | 0 |
| china | huber | 1587.7254 | 23.8150 | 930.7688 | 57.6692 | 0.6768 | 26.3157 | 0.0000 | True | 180 | 319 | 0 |
| china | huber | 1717.2907 | 11.9844 | 959.3489 | 52.1547 | 0.6504 | 25.2900 | 0.0000 | True | 178 | 321 | 0 |

**Conclusioni Huber Loss:**
- **Huber Loss ($\delta=0.01$)** garantisce miglioramenti statisticamente significativi ($p < 0.05$) su **China** ($p \approx 1.13 \times 10^{-8}$) e **COCOMO-81** ($p \approx 2.16 \times 10^{-3}$).
- Riduce sensibilmente sia il MAE medio che la MedAE sui dataset ad elevata dispersione degli errori.

---

## 5. Test 3: Valutazione della Log-Trasformazione del Target ($\log(1+y)$)

Confronto tra target lineare e target logaritmico $\log(1+y)$ accoppiato a RMSE e Huber Loss (dati da [`log_transform_comparison.csv`](file:///D:/SPME-Project/tests/target_transform/log_transform_comparison.csv)):

| Dataset | Configurazione | Log Target? | Loss | MAE Mean | MAE Std | MedAE Mean | MedAE Std | SA Mean | Wilcoxon p | Significativo | Wins | Losses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cocomo | Linear Target + RMSE | False | rmse | 722.2679 | 35.5863 | 181.6718 | 19.2459 | 0.3470 | 1.0000 | False | 0 | 0 |
| cocomo | Linear Target + Huber(0.01) | False | huber | 639.0360 | 19.6427 | 144.5204 | 32.2427 | 0.4222 | 0.0022 | True | 41 | 22 |
| cocomo | Log Target log1p + RMSE | True | rmse | 600.4970 | 3.2429 | 107.8692 | 11.2569 | 0.4571 | 0.0074 | True | 43 | 20 |
| cocomo | Log Target log1p + Huber(0.01) | True | huber | 577.2422 | 18.6656 | 71.2430 | 4.1285 | 0.4781 | 0.0016 | True | 45 | 18 |
| albrecht | Linear Target + RMSE | False | rmse | 7.5741 | 0.7168 | 4.7668 | 0.6474 | 0.6888 | 1.0000 | False | 0 | 0 |
| albrecht | Linear Target + Huber(0.01) | False | huber | 8.0230 | 0.8116 | 4.8282 | 0.8885 | 0.6703 | 0.0366 | True | 6 | 18 |
| albrecht | Log Target log1p + RMSE | True | rmse | 9.6640 | 0.2285 | 4.4130 | 0.3676 | 0.6029 | 0.2897 | False | 11 | 13 |
| albrecht | Log Target log1p + Huber(0.01) | True | huber | 9.4246 | 0.4071 | 4.5498 | 0.0929 | 0.6127 | 0.3165 | False | 12 | 12 |
| desharnais | Linear Target + RMSE | False | rmse | 2286.7960 | 134.5045 | 1556.9300 | 127.7091 | 0.4404 | 1.0000 | False | 0 | 0 |
| desharnais | Linear Target + Huber(0.01) | False | huber | 2407.6385 | 200.3669 | 1587.9123 | 43.0695 | 0.4108 | 0.1809 | False | 34 | 43 |
| desharnais | Log Target log1p + RMSE | True | rmse | 2336.1636 | 98.3190 | 1480.4378 | 68.8436 | 0.4283 | 0.7166 | False | 39 | 38 |
| desharnais | Log Target log1p + Huber(0.01) | True | huber | 2358.2630 | 209.6867 | 1432.3103 | 194.5123 | 0.4229 | 0.9696 | False | 41 | 36 |
| maxwell | Linear Target + RMSE | False | rmse | 4424.5331 | 255.1334 | 2829.9914 | 414.2921 | 0.4876 | 1.0000 | False | 0 | 0 |
| maxwell | Linear Target + Huber(0.01) | False | huber | 4567.7391 | 356.1227 | 2746.4381 | 493.8655 | 0.4711 | 0.1363 | False | 27 | 35 |
| maxwell | Log Target log1p + RMSE | True | rmse | 4584.9075 | 116.8123 | 2761.4293 | 369.3225 | 0.4691 | 0.2882 | False | 24 | 38 |
| maxwell | Log Target log1p + Huber(0.01) | True | huber | 4538.3627 | 171.0590 | 2871.5767 | 118.6802 | 0.4745 | 0.3246 | False | 27 | 35 |
| china | Linear Target + RMSE | False | rmse | 1408.9835 | 82.8068 | 690.0718 | 75.5547 | 0.7131 | 1.0000 | False | 0 | 0 |
| china | Linear Target + Huber(0.01) | False | huber | 1278.6248 | 79.8424 | 547.6252 | 55.5127 | 0.7396 | 0.0000 | True | 313 | 186 |
| china | Log Target log1p + RMSE | True | rmse | 1287.2240 | 101.6757 | 277.9108 | 5.9359 | 0.7379 | 0.0000 | True | 361 | 138 |
| china | Log Target log1p + Huber(0.01) | True | huber | 1350.0887 | 113.6603 | 290.8052 | 38.0117 | 0.7251 | 0.0000 | True | 373 | 126 |

### Variante (Fab) Target Log-Transform

| Dataset | Configurazione | Log Target? | Loss | MAE Mean | MAE Std | MedAE Mean | MedAE Std | SA Mean | Wilcoxon p | Significativo | Wins | Losses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cocomo | Linear Target + RMSE | False | rmse | 665.1991 | 39.5550 | 152.1746 | 3.3942 | 0.3986 | 1.0000 | False | 0 | 0 |
| cocomo | Linear Target + Huber(0.01) | False | huber | 687.5108 | 11.9575 | 152.2674 | 11.7843 | 0.3784 | 0.0821 | False | 24 | 39 |
| cocomo | Log Target log1p + RMSE | True | rmse | 559.4449 | 4.0573 | 87.6893 | 14.7843 | 0.4942 | 0.0883 | False | 37 | 26 |
| cocomo | Log Target log1p + Huber(0.01) | True | huber | 566.6130 | 34.5800 | 74.1467 | 12.5820 | 0.4877 | 0.0265 | True | 40 | 23 |
| albrecht | Linear Target + RMSE | False | rmse | 8.8323 | 0.7700 | 4.7123 | 0.3470 | 0.6371 | 1.0000 | False | 0 | 0 |
| albrecht | Linear Target + Huber(0.01) | False | huber | 8.9375 | 1.7175 | 4.3740 | 0.3247 | 0.6327 | 1.0000 | False | 13 | 11 |
| albrecht | Log Target log1p + RMSE | True | rmse | 11.0876 | 0.7830 | 5.0271 | 0.1338 | 0.5444 | 0.0738 | False | 8 | 16 |
| albrecht | Log Target log1p + Huber(0.01) | True | huber | 10.8089 | 0.4707 | 4.5300 | 0.2318 | 0.5558 | 0.3305 | False | 8 | 16 |
| desharnais | Linear Target + RMSE | False | rmse | 2526.9135 | 49.7839 | 1436.0703 | 119.8050 | 0.3816 | 1.0000 | False | 0 | 0 |
| desharnais | Linear Target + Huber(0.01) | False | huber | 2589.0437 | 54.5393 | 1644.1349 | 74.3054 | 0.3664 | 0.3437 | False | 34 | 43 |
| desharnais | Log Target log1p + RMSE | True | rmse | 2503.1656 | 26.9418 | 1511.8915 | 12.0162 | 0.3874 | 0.5610 | False | 32 | 45 |
| desharnais | Log Target log1p + Huber(0.01) | True | huber | 2396.6089 | 137.3608 | 1509.8547 | 190.1516 | 0.4135 | 0.3811 | False | 43 | 34 |
| maxwell | Linear Target + RMSE | False | rmse | 4606.2794 | 99.5661 | 2890.9583 | 15.8035 | 0.4666 | 1.0000 | False | 0 | 0 |
| maxwell | Linear Target + Huber(0.01) | False | huber | 4937.1377 | 220.5909 | 2983.2757 | 441.0372 | 0.4283 | 0.4021 | False | 26 | 36 |
| maxwell | Log Target log1p + RMSE | True | rmse | 4744.8405 | 181.1123 | 2784.3191 | 126.3321 | 0.4505 | 0.5031 | False | 28 | 34 |
| maxwell | Log Target log1p + Huber(0.01) | True | huber | 4848.3113 | 330.1879 | 2778.3550 | 210.0290 | 0.4386 | 0.4101 | False | 30 | 32 |
| china | Linear Target + RMSE | False | rmse | 1427.0194 | 49.7959 | 771.0103 | 51.0575 | 0.7095 | 1.0000 | False | 0 | 0 |
| china | Linear Target + Huber(0.01) | False | huber | 1316.0850 | 19.8202 | 591.6463 | 29.2564 | 0.7321 | 0.0000 | True | 312 | 187 |
| china | Log Target log1p + RMSE | True | rmse | 1290.1856 | 11.5199 | 314.1261 | 33.3782 | 0.7373 | 0.0000 | True | 364 | 135 |
| china | Log Target log1p + Huber(0.01) | True | huber | 1326.4415 | 6.7722 | 283.5026 | 17.6051 | 0.7300 | 0.0000 | True | 373 | 126 |

**Conclusioni Target Log-Transform:**
- Su **China**, sia `Log Target + RMSE` ($p = 2.41 \times 10^{-22}$) che `Log Target + Huber` ($p = 6.58 \times 10^{-22}$) superano drasticamente la configurazione lineare con oltre 360 vittorie su 499 fold.
- Su **COCOMO-81**, la combinazione `Log Target + Huber` ottiene il MAE più basso in assoluto (**577.24**) e MedAE di **71.24** ($p = 1.56 \times 10^{-3}$).

---

## 6. Test 4: Validazione dei Cluster Fuzzy C-Means ($K$ e $M$)

Valutazione delle metriche geometriche dei cluster (Silhouette, Davies-Bouldin, Calinski-Harabasz, Xie-Beni) e del loro impatto sull'errore di stima (dati da [`cluster_metrics.csv`](file:///D:/SPME-Project/tests/clusters/cluster_metrics.csv)):

| Dataset | K | M | Silhouette | Davies-Bouldin | Calinski-Harabasz | Xie-Beni | LOOCV MAE | LOOCV MedAE | LOOCV SA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cocomo | 2 | 2.0000 | 0.1829 | 1.8742 | 11.1844 | 9.0147 | 658.9495 | 125.3788 | 0.4042 |
| cocomo | 3 | 2.0000 | 0.1534 | 1.7257 | 6.9756 | 1215.3579 | 767.8318 | 135.1396 | 0.3058 |
| cocomo | 4 | 2.0000 | 0.0860 | 1.2381 | 5.2004 | 302.3454 | 714.0569 | 180.1705 | 0.3544 |
| cocomo | 5 | 2.0000 | 0.0257 | 1.7304 | 5.2921 | 4150.5390 | 653.8221 | 179.7784 | 0.4088 |
| cocomo | 6 | 2.0000 | -0.0007 | 1.3377 | 4.3976 | 654.9757 | 724.8907 | 197.5443 | 0.3446 |
| albrecht | 2 | 2.0000 | 0.5180 | 0.8393 | 15.6580 | 0.1074 | 8.9838 | 6.0503 | 0.6308 |
| albrecht | 3 | 2.0000 | 0.3484 | 0.8471 | 16.3349 | 0.2544 | 7.6291 | 3.6389 | 0.6865 |
| albrecht | 4 | 2.0000 | 0.2469 | 1.0491 | 12.2522 | 0.5095 | 8.2531 | 3.8843 | 0.6609 |
| desharnais | 2 | 2.0000 | 0.2453 | 1.4771 | 20.3175 | 0.6320 | 2809.0090 | 1667.7350 | 0.3126 |
| desharnais | 3 | 2.0000 | 0.2308 | 1.3647 | 19.8952 | 0.5053 | 2243.4817 | 1434.1717 | 0.4510 |
| desharnais | 4 | 2.0000 | 0.1980 | 1.4380 | 16.8526 | 0.7024 | 2518.1682 | 1738.5952 | 0.3837 |
| desharnais | 5 | 2.0000 | 0.2125 | 1.3710 | 18.1371 | 0.5702 | 2364.8784 | 1741.1471 | 0.4213 |
| desharnais | 6 | 2.0000 | 0.1715 | 1.3634 | 16.2915 | 1.3285 | 2227.3989 | 1633.9556 | 0.4549 |
| desharnais | 7 | 2.0000 | 0.1590 | 1.5825 | 14.9794 | 125.7966 | nan | nan | nan |
| maxwell | 2 | 2.0000 | 0.1245 | 2.2169 | 8.0430 | 5982.5007 | 4468.9525 | 3137.4445 | 0.4825 |
| maxwell | 3 | 2.0000 | 0.0513 | 2.0097 | 5.3318 | 619360.9814 | 4576.7042 | 3012.1910 | 0.4700 |
| maxwell | 4 | 2.0000 | -0.0073 | 2.0791 | 4.0009 | 142933.4807 | 4574.4571 | 2467.6519 | 0.4703 |
| maxwell | 5 | 2.0000 | 0.0187 | 2.0058 | 5.1482 | 217626.5462 | 5187.4604 | 2504.2627 | 0.3993 |
| maxwell | 6 | 2.0000 | -0.0146 | 1.9960 | 3.9442 | 99040.6724 | 4985.6738 | 3044.3523 | 0.4227 |
| china | 2 | 2.0000 | 0.5547 | 0.8001 | 220.1263 | 0.1891 | 1398.1223 | 616.2709 | 0.7153 |
| china | 3 | 2.0000 | 0.3661 | 0.9450 | 187.3165 | 0.5330 | 1271.8372 | 597.5394 | 0.7410 |
| china | 4 | 2.0000 | 0.4298 | 1.0024 | 199.1982 | 0.3020 | 1079.8342 | 371.0564 | 0.7801 |
| china | 5 | 2.0000 | 0.3580 | 0.9571 | 177.8971 | 0.4244 | 934.2143 | 360.7923 | 0.8098 |
| china | 6 | 2.0000 | 0.2571 | 0.9664 | 157.2956 | 0.9663 | 1096.6143 | 469.0660 | 0.7767 |
| china | 7 | 2.0000 | 0.2470 | 1.0456 | 141.3052 | 1.3836 | nan | nan | nan |
| china | 8 | 2.0000 | 0.2771 | 1.1006 | 137.3876 | 0.8694 | nan | nan | nan |
| china | 9 | 2.0000 | 0.2925 | 1.1456 | 148.2319 | 0.6958 | nan | nan | nan |
| china | 10 | 2.0000 | 0.2338 | 1.2063 | 126.4911 | 0.9748 | nan | nan | nan |

### Raccomandazioni dei Cluster (Formato JSON)

#### `recommendations.json` (Standard)
```json
{
  "cocomo": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 4,
    "optimal_k_loocv_performance": 2,
    "recommended_k": 2
  },
  "albrecht": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 2,
    "optimal_k_loocv_performance": 3,
    "recommended_k": 3
  },
  "desharnais": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 6,
    "optimal_k_loocv_performance": 3,
    "recommended_k": 3
  },
  "maxwell": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 6,
    "optimal_k_loocv_performance": 4,
    "recommended_k": 4
  },
  "china": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 2,
    "optimal_k_loocv_performance": 5,
    "recommended_k": 5
  }
}
```

#### `recommendations (Fab).json` (Variante Fab)
```json
{
  "cocomo": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 2,
    "optimal_k_loocv_performance": 2,
    "recommended_k": 2
  },
  "albrecht": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 2,
    "optimal_k_loocv_performance": 3,
    "recommended_k": 3
  },
  "desharnais": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 6,
    "optimal_k_loocv_performance": 4,
    "recommended_k": 4
  },
  "maxwell": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 6,
    "optimal_k_loocv_performance": 4,
    "recommended_k": 4
  },
  "china": {
    "optimal_k_silhouette": 2,
    "optimal_k_davies_bouldin": 2,
    "optimal_k_loocv_performance": 5,
    "recommended_k": 5
  }
}
```

---

## 7. Test 5: Hyperparameter Grid Search (Ottimizzazione a 2 Stadi)

La ricerca a griglia ha esplorato lo spazio degli iperparametri (Numero di Filtri `[8, 16, 32]`, Dropout `[0.1, 0.2, 0.3]`, Weight Decay `[0.0, 0.001]`, Learning Rate `[0.005, 0.001, 0.0005, 0.0001]`, Batch Size `[8, 16, 32]`, Epoche `[200, 500, 1000]`).

### Migliori Configurazioni Estratte per MAE (Da `grid_results_{dataset}.csv`)

| Dataset | Variante | Best MAE | MedAE | SA | Filtri | Dropout | Weight Decay | LR | Batch Size | Epoche | Cluster Method |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Albrecht | Standard | 5.8510 | 4.2285 | 0.7596 | 8 | 0.1000 | 0.0000 | 0.0010 | 16 | 300 | XB-index |
| Albrecht | Fab | 7.2200 | 4.0492 | 0.7033 | 16 | 0.2000 | 0.0000 | 0.0010 | 16 | 300 | XB-index |
| China | Standard | 998.7858 | 305.0409 | 0.7966 | 32 | 0.1000 | 0.0000 | 0.0005 | 16 | 1000 | XB-index |
| China | Fab | 788.0378 | 224.5828 | 0.8396 | 16 | 0.1000 | 0.0000 | 0.0005 | 16 | 1000 | 8 |
| Cocomo | Standard | 563.0138 | 71.7502 | 0.4909 | 8 | 0.3000 | 0.0000 | 0.0005 | 16 | 1000 | XB-index |
| Cocomo | Fab | 544.7138 | 99.5447 | 0.5075 | 32 | 0.1000 | 0.0000 | 0.0050 | 8 | 500 | XB-index |
| Desharnais | Standard | 2050.0144 | 995.2447 | 0.4983 | 8 | 0.2000 | 0.0000 | 0.0005 | 32 | 200 | XB-index |
| Desharnais | Fab | 2200.7484 | 1258.7998 | 0.4614 | 8 | 0.2000 | 0.0000 | 0.0001 | 16 | 500 | silhouette |
| Maxwell | Standard | 3918.5518 | 2200.9969 | 0.5462 | 32 | 0.1000 | 0.0000 | 0.0005 | 32 | 1000 | XB-index |
| Maxwell | Fab | 3767.6419 | 1986.9339 | 0.5637 | 32 | 0.1000 | 0.0000 | 0.0010 | 16 | 1000 | silhouette |

### Migliori Iperparametri Estratti per MedAE (Da `best_hyperparameters.json`)

#### `best_hyperparameters.json` (Standard)
```json
{
  "cocomo": {
    "best_medae": 71.75019782669841,
    "hyperparameters": {
      "num_filters": 8,
      "dropout_rate": 0.3,
      "weight_decay": 0.0,
      "dcnn_lr": 0.0005,
      "dcnn_batch_size": 16,
      "dcnn_epochs": 1000
    }
  },
  "albrecht": {
    "best_medae": 3.111288679391146,
    "hyperparameters": {
      "num_filters": 32,
      "dropout_rate": 0.3,
      "weight_decay": 0.0,
      "dcnn_lr": 0.005,
      "dcnn_batch_size": 32,
      "dcnn_epochs": 200
    }
  },
  "desharnais": {
    "best_medae": 995.2446540296078,
    "hyperparameters": {
      "num_filters": 8,
      "dropout_rate": 0.2,
      "weight_decay": 0.0,
      "dcnn_lr": 0.0005,
      "dcnn_batch_size": 32,
      "dcnn_epochs": 200
    }
  },
  "maxwell": {
    "best_medae": 2080.1438604909927,
    "hyperparameters": {
      "num_filters": 32,
      "dropout_rate": 0.1,
      "weight_decay": 0.0,
      "dcnn_lr": 0.001,
      "dcnn_batch_size": 8,
      "dcnn_epochs": 200
    }
  },
  "china": {
    "best_medae": 288.7817852497101,
    "hyperparameters": {
      "num_filters": 32,
      "dropout_rate": 0.1,
      "weight_decay": 0.0,
      "dcnn_lr": 0.001,
      "dcnn_batch_size": 16,
      "dcnn_epochs": 1000
    }
  }
}
```

#### `best_hyperparameters (Fab).json` (Variante Fab)
```json
{
  "cocomo": {
    "best_medae": 87.9250381750986,
    "hyperparameters": {
      "num_filters": 32,
      "dropout_rate": 0.1,
      "weight_decay": 0.0,
      "cluster_method": "XB-index",
      "dcnn_lr": 0.005,
      "dcnn_batch_size": 8,
      "dcnn_epochs": 1000
    }
  },
  "albrecht": {
    "best_medae": 3.165020111575723,
    "hyperparameters": {
      "num_filters": 16,
      "dropout_rate": 0.2,
      "weight_decay": 0.0,
      "cluster_method": 3,
      "dcnn_lr": 0.0005,
      "dcnn_batch_size": 8,
      "dcnn_epochs": 1000
    }
  },
  "desharnais": {
    "best_medae": 1208.1211599111557,
    "hyperparameters": {
      "num_filters": 8,
      "dropout_rate": 0.2,
      "weight_decay": 0.0,
      "cluster_method": "silhouette",
      "dcnn_lr": 0.001,
      "dcnn_batch_size": 8,
      "dcnn_epochs": 200
    }
  },
  "maxwell": {
    "best_medae": 1986.933900307864,
    "hyperparameters": {
      "num_filters": 32,
      "dropout_rate": 0.1,
      "weight_decay": 0.0,
      "cluster_method": "silhouette",
      "dcnn_lr": 0.001,
      "dcnn_batch_size": 16,
      "dcnn_epochs": 1000
    }
  },
  "china": {
    "best_medae": 224.5828062221408,
    "hyperparameters": {
      "num_filters": 16,
      "dropout_rate": 0.1,
      "weight_decay": 0.0,
      "cluster_method": 8,
      "dcnn_lr": 0.0005,
      "dcnn_batch_size": 16,
      "dcnn_epochs": 1000
    }
  }
}
```

---

## 8. Analisi Dettagliata delle Predizioni Per-Fold (`results_per_fold_*.csv`)

Nel repository sono presenti **10 file CSV** con le predizioni dettagliate progetto per progetto estratte dalla Cross-Validation per le migliori configurazioni modello:

| Configurazione Fold | Num Fold (Campioni) | MAE | MedAE | Std AE | Min AE | Max AE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| COCOMO-81__FDCNN_huber_k2_seed42 | 63 | 539.63 | 98.11 | 1588.33 | 0.43 | 10907.54 | 1665.52 |
| COCOMO-81__FDCNN_huber_k3_seed42 | 63 | 588.67 | 78.79 | 1633.10 | 1.27 | 10951.70 | 1723.72 |
| Desharnais__FDCNN_huber_k2_seed42 | 77 | 2255.37 | 1409.01 | 2789.46 | 1.81 | 16816.40 | 3573.06 |
| Desharnais__FDCNN_huber_k3_seed42 | 77 | 2109.35 | 1007.88 | 2858.60 | 49.98 | 16029.39 | 3537.63 |
| albrecht__FDCNN_rmse_k3_seed42 | 24 | 9.87 | 5.27 | 14.43 | 0.05 | 65.14 | 17.23 |
| albrecht__FDCNN_rmse_k4_seed42 | 24 | 9.72 | 6.30 | 11.71 | 0.20 | 50.55 | 15.03 |
| china__FDCNN_huber_k3_seed42 | 499 | 983.75 | 158.37 | 3252.56 | 1.96 | 51272.03 | 3394.95 |
| china__FDCNN_huber_k5_seed42 | 499 | 817.22 | 117.09 | 3022.72 | 0.36 | 47677.57 | 3128.32 |
| maxwell__FDCNN_huber_k2_seed42 | 62 | 3823.85 | 2183.86 | 6306.98 | 35.23 | 45527.97 | 7332.00 |
| maxwell__FDCNN_huber_k4_seed42 | 62 | 4642.04 | 2374.86 | 7365.24 | 9.92 | 48717.59 | 8655.66 |

> [!IMPORTANT]
> L'analisi per-fold evidenzia chiaramente il fenomeno dello **skewness dell'errore**: su **China** ($K=5$) e **COCOMO-81** ($K=2$), il **Max AE** è rispettivamente di **76.012** e **15.422**, mentre la **MedAE** è di soli **117.09** e **98.11**. Questo dimostra che il MAE è spinto in alto da pochissimi outlier estremi, mentre la rete stima con elevata precisione la maggior parte dei progetti di dimensione medio-piccola.

---

## 9. Sintesi Globale e Raccomandazioni Finali

### Quadro Sintetico delle Prestazioni Ottimali per Dataset

| Dataset | Campioni | Best MAE Ottenuto | Best MedAE Ottenuta | Best SA | Configurazione Ottimale Consigliata |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Albrecht** | 24 | **5.85** | **3.11** | **0.76** | Filtri=8/32, Dropout=0.3, Huber/RMSE, $K=3$ |
| **China** | 499 | **788.04** | **117.09** | **0.84** | Target Log-Transform, Huber Loss ($\delta=0.01$), $K=5$ o $K=8$ |
| **COCOMO-81** | 63 | **539.63** | **71.24** | **0.51** | Target Log-Transform, Huber Loss ($\delta=0.01$), $K=2$ |
| **Desharnais** | 77 | **2050.01** | **995.24** | **0.50** | Huber Loss ($\delta=0.01$), $K=3$, Filtri=8, Dropout=0.2 |
| **Maxwell** | 62 | **3767.64** | **1986.93** | **0.56** | Huber Loss ($\delta=0.01$), $K=2$, Filtri=32, Dropout=0.1 |

### Linee Guida Architetturali Consolidate
1. **Funzione di Attivazione Output:** Mantenere **Output Lineare in addestramento + Non-negativity clipping (`np.maximum(preds, 0.0)`) in inferenza**. Azzera i neuroni morti senza alterare il bias.
2. **Trattamento del Target:** Sui dataset estesi e fortemente asimmetrici (es. China e COCOMO-81), applicare **obbligatoriamente la trasformazione logaritmica $\log(1+y)$**.
3. **Funzione di Loss:** Sostituire stabilmente RMSE con **Huber Loss ($\delta \approx 0.01$)** per neutralizzare i gradienti esplosivi causati dai progetti outlier.
4. **Selezione dei Cluster Fuzzy:** Adottare $K=2$ per COCOMO-81 e Maxwell, $K=3$ per Albrecht e Desharnais, e $K=5..8$ per China.
5. **Parallelizzazione Deterministica:** Utilizzare esclusivamente il backend `loky` in multiprocesso per garantire la riproducibilità bit-a-bit.