# Piano di Test Ottimizzato per Fuzzy-DCNN

## Panoramica

Il presente documento definisce il piano di sperimentazione ottimizzato per la validazione e il tuning dell'architettura **Fuzzy-DCNN** per l'esigenza di *Software Development Effort Estimation* (SDEE). 

L'ottimizzazione del piano sintetizza i principi teorici ed empirici estratti dalla letteratura scientifica recente (2023–2026) presente in `sample-papers/` (tra cui TabNet+HHO, Stacking Ensembles SELI/SENSE, LSTM e CNN per SDEE) e si integra nativamente con la pipeline di validazione del progetto (`fuzzy-DCNN-implementation/evaluate.py`).

---

## Architettura dei Test e Struttura Directory

La suite di test è organizzata nella cartella `tests/` con codice modulare e riusabile:

* `tests/utils.py`: Funzioni di utility condivise (loader dati, calcolo metriche LOOCV/10-fold, test statistici).
* `tests/clusters/`: Test 1 – Selezione e validazione del numero di cluster fuzzy $k$ e parametro $m$.
* `tests/deadneurons/`: Test 2 – Analisi delle funzioni di attivazione e mitigazione dei neuroni morti.
* `tests/gridsearch/`: Test 3 – Ottimizzazione iperparametrica efficiente a 2 stadi / Bayesiana.
* `tests/huber_loss/`: Test 4 – Valutazione della Huber Loss e taratura del parametro $\delta$.
* `tests/target_transform/`: Test 5 (Nuovo SOTA) – Impatto della trasformazione logaritmica del target ($\log(1+y)$).
* `tests/baselines/`: Test 6 – Confronto con i modelli baseline classici (Di Martino SLR, Random Forest, SVM/SVR).

---

## Standard Globali di Esecuzione e Valutazione

Tutti i test dovranno rispettare rigorosamente i seguenti requisiti metodologici:

1. **Protocollo di Validazione Rigoroso:**
   * **LOOCV (Leave-One-Out Cross-Validation):** Obbligatorio per i dataset di piccole dimensioni (*COCOMO-81*, *Albrecht*, *Desharnais*, *Maxwell*, *Kemerer*).
   * **10-Fold Cross-Validation:** Utilizzato per il dataset *China* (campione più numeroso) con seed di splitting controllato.
   * *N.B.:* È vietato l'uso di uno split statico 70/30 nei test principali, per evitare alta varianza e conclusioni fuorvianti su dataset di 24-60 campioni.

2. **Set di Metriche di Valutazione SOTA:**
   * **MAE (Mean Absolute Error):** Errore medio assoluto.
   * **MedAE (Median Absolute Error):** Errore mediano assoluto (robusto contro i progetti outlier).
   * **SA (Standardized Accuracy):** $1 - \frac{\text{MAE}}{\text{MAE}_{\text{random}}}$, dove $\text{MAE}_{\text{random}}$ è calcolato su 1000 permutazioni casuali.
   * **Effect Size ($\Delta$):** Dimensione dell'effetto rispetto al baseline casuale.
   * **Tasso Neuroni Morti (% Zero Preds):** Percentuale di predizioni esattamente pari a `target_min` (0.0).

3. **Validazione Statistica:**
   * Ogni configurazione viene eseguita su **5 seed casuali** differenti (es. 42, 123, 456, 789, 999).
   * Calcolo del **Test dei ranghi con segno di Wilcoxon** ($p < 0.05$) per confermare che i miglioramenti rispetto al baseline siano statisticamente significativi.

---

## Test 1: Validazione dei Cluster Fuzzy (`tests/clusters/`)

### Obiettivo
Sostituire l'indice Xie-Beni (XB), che presenta instabilità numerica e tendenza a selezionare cluster degeneri con feature scalate, con tre metriche complementari per determinare il $k$ ottimale ($k \in [2, 10]$) e valutare l'effetto dell'esponente di fuzzificazione $m \in [1.5, 2.0, 2.5]$:
1. **Silhouette Score:** Misura la coesione e separazione dei cluster (valori più alti indicano cluster meglio definiti).
2. **Davies-Bouldin Index (DB):** Misura la similarità media tra ogni cluster e il suo più simile (valori più bassi indicano migliore separazione).
3. **Calinski-Harabasz Index (CH):** Rapporto tra la dispersione tra cluster e la dispersione intra-cluster (valori più alti sono migliori).

### Piano di Esecuzione
1. Convertire le appartenenze fuzzy risultanti da Fuzzy C-Means (FCM) in etichette hard tramite `argmax`.
2. Calcolare Silhouette, DB e CH su tutti i dataset per $k \in [2, 10]$ e $m \in [1.5, 2.0, 2.5]$.
3. Eseguire una prova end-to-end con LOOCV per verificare se il $k$ raccomandato dalle metriche geometriche corrisponde al $k$ che minimizza la MedAE/MAE dell'FDCNN.
4. Salvare i risultati in `tests/clusters/cluster_metrics.csv` e `tests/clusters/recommendations.json`.

---

## Test 2: Analisi dei Neuroni Morti e Attivazioni Output (`tests/deadneurons/`)

### Obiettivo
Identificare l'attivazione/strategia di output ottimale per eliminare il fenomeno dei neuroni morti ("dying ReLU") senza introdurre distorsioni nella stima dell'effort. 

### Varianti del Modello
* **Variante A (ReLU Standard):** `nn.ReLU()` al layer di output (rischio di azzeramento permanente dei gradienti).
* **Variante B (Softplus):** `nn.Softplus()` al layer di output (gradiente continuo, ma possibile bias positivo nei valori piccoli).
* **Variante C (Linear + Clipping in Inferenza - Corrente):** Layer lineare durante il training + `np.maximum(preds, 0.0)` in inferenza.
* **Variante D (LeakyReLU + Clipping):** `nn.LeakyReLU(negative_slope=0.01)` al layer di output per mantenere gradiente per input negativi.
* **Variante E (ELU + 1):** `nn.ELU() + 1.0` per garantire uscite non negative con gradiente fluido.

### Piano di Esecuzione
1. Addestrare tutte e 5 le varianti sui dataset *China* e *Desharnais* (i più soggetti a skewness e neuroni morti) sotto 10-fold CV / LOOCV su 5 seed.
2. Tracciare il **Dead Neuron Rate** (% di predizioni bloccate a 0.0), il gradiente medio dei layer FC ed il MAE/MedAE/SA.
3. Salvare i risultati in `tests/deadneurons/activation_comparison.csv`.

---

## Test 3: Ottimizzazione Iperparametrica Efficiente (`tests/gridsearch/`)

### Obiettivo
Trovare la configurazione ottimale degli iperparametri per ciascun dataset, superando i valori di default fissi. Per evitare l'esplosione combinatoria ($432$ combinazioni $\times$ LOOCV $\approx$ ore di calcolo), si adotta una **ricerca a 2 stadi** (o ricerca Bayesiana via Optuna).

### Spazio di Ricerca

| Iperparametro | Valori di Ricerca | Default Attuale |
| --- | --- | --- |
| **Learning Rate (lr)** | 0.005, 0.001, 0.0005, 0.0001 | 0.001 |
| **Batch Size** | 8, 16, 32 | 16 |
| **Epoche** | 200, 500, 1000 | 1000 |
| **Weight Decay** | 0.0, 0.001, 0.0001 | 0.0 |
| **Dropout Rate** | 0.1, 0.2, 0.3 | 0.2 |
| **Filtri Conv (num_filters)** | 8, 16, 32 | 16 |

### Piano di Esecuzione (Strategia a 2 Stadi)
* **Stadio 1 (Fuzzy & Architettura):** Fissare lr=0.001 e trovare la combinazione migliore di `num_filters`, `dropout_rate` e `weight_decay`.
* **Stadio 2 (Training & Opt):** Mantenere l'architettura ottimale dallo Stadio 1 e ottimizzare `lr`, `batch_size` ed `epoche` (con Early Stopping se la loss non migliora per 50 epoche).
* Eseguire la ricerca sotto LOOCV / 10-fold CV e registrare i risultati in `tests/gridsearch/grid_results_{dataset}.csv` e la summary finale in `best_hyperparameters.json`.

---

## Test 4: Valutazione Huber Loss e Loss Robuste (`tests/huber_loss/`)

### Obiettivo
Valutare l'efficacia della **Huber Loss (Smooth L1)** rispetto alla **RMSELoss** classica e alla **MAE Loss** per gestire la presenza di outlier e la distribuzione skewed dell'effort software.

### Parametri e Varianti Loss
* **RMSE Loss:** Loss standard (penalità quadratica $e^2$, molto sensibile agli outlier).
* **MAE Loss:** Penalità lineare $|e|$ (robusta, ma con derivata non continua a 0).
* **Huber Loss ($\delta$):** Valori di $\delta \in [0.01, 0.05, 0.1, 0.2, 0.5]$. (Poiché il target è scalato in $[0, 1]$, delta piccoli regolano la transizione quadratica/lineare).

### Piano di Esecuzione
1. Eseguire l'addestramento su tutti i dataset registrati con le 7 varianti di loss under LOOCV/10-fold CV su 5 seed.
2. Analizzare l'impatto congiunto su MAE, MedAE, SA e deviazione standard dell'errore tra i fold.
3. Dimostrare empiricamente che la Huber Loss riduce il MAE senza peggiorare la MedAE rispetto a RMSELoss.
4. Salvare i risultati in `tests/huber_loss/huber_results.csv`.

---

## Test 5 (Nuovo SOTA): Trasformazione del Target ($\log(1+y)$) (`tests/target_transform/`)

### Obiettivo
Testare l'impatto della trasformazione logaritmica della variabile target $y_{\text{log}} = \log(1 + y)$ durante il training e de-trasformazione $\hat{y} = \exp(\hat{y}_{\text{log}}) - 1$ durante l'inferenza. La letteratura evidenzia che nei dataset SDEE la trasformazione logaritmica stabilizza la varianza e riduce drasticamente l'impatto degli outlier.

### Piano di Esecuzione
1. Confrontare il modello baseline (scala lineare min-max) con il modello log-trasformato (`use_log_target=True`).
2. Valutare se la combinazione **Target Log-Transform + Huber Loss** garantisce la massima Standardized Accuracy (SA) e stabilità su tutti i dataset.
3. Salvare i risultati in `tests/target_transform/log_transform_comparison.csv`.

---

## Test 6: Confronto con i Modelli Baseline della Letteratura (`tests/baselines/`)

### Obiettivo
Confrontare le prestazioni del modello ottimizzato **Fuzzy-DCNN** con i tre modelli baseline storici e consolidati nella letteratura scientifica di SDEE:
1. **Di Martino et al. (2020):** Simple Linear Regression (SLR) su metriche Function Points / LOC.
2. **Random Forest (RF):** Regressione ensemble non lineare basata su foreste decisionali.
3. **Support Vector Regression (SVR / SVM):** Regressione a iperpiani con kernel radiale (RBF).

### Piano di Esecuzione
1. Eseguire la comparazione sui 5 dataset sotto LOOCV (*COCOMO-81*, *Albrecht*, *Desharnais*, *Maxwell*) e 10-Fold CV (*China*).
2. Valutare MAE, MedAE e Standardized Accuracy (SA) per ciascun modello e dataset.
3. Dimostrare che Fuzzy-DCNN supera i baseline convenzionali sull'errore mediano e sulla stabilità rispetto agli outlier.
4. Salvare i risultati in `tests/baselines/baseline_comparison_results.csv`.

---

## Tabella Riassuntiva dei Test e Deliverable

| ID Test | Nome Test | Variabili Chiave | File Output |
| --- | --- | --- | --- |
| **Test 1** | Cluster Validation | $k \in [2, 10]$, $m \in [1.5, 2.5]$, Silhouette, DB, CH | `tests/clusters/cluster_metrics.csv` |
| **Test 2** | Dead Neurons | ReLU, Softplus, Linear+Clip, LeakyReLU, ELU | `tests/deadneurons/activation_comparison.csv` |
| **Test 3** | Grid Search | LR, Batch, Epoche, Weight Decay, Filters, Dropout | `tests/gridsearch/grid_results_{dataset}.csv` |
| **Test 4** | Huber Loss | RMSE vs MAE vs Huber ($\delta \in [0.01..0.5]$) | `tests/huber_loss/huber_results.csv` |
| **Test 5** | Target Transform | Linear Target vs Log Target ($\log(1+y)$) | `tests/target_transform/log_transform_comparison.csv` |
| **Test 6** | Baseline Models | Di Martino SLR, Random Forest, SVR vs Fuzzy-DCNN | `tests/baselines/baseline_comparison_results.csv` |

---

## Conclusione e Prossimi Passi

Attraverso questo piano di test ottimizzato, le sperimentazioni forniranno risultati riproducibili, statisticamente validati ed allineati ai requisiti delle riviste e conferenze internazionali di Software Engineering (IEEE TSE, ACM TOSEM, EMSE).