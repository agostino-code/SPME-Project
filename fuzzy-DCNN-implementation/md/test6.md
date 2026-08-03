# Report Fase 7A: Valutazione PCA (Principal Component Analysis)

Come concordato, è stata implementata e testata la riduzione di dimensionalità tramite PCA (95% di varianza mantenuta) in `fold_worker.py`. I risultati sono stati ottenuti mediando 3 seed iniziali per ciascun dataset, confrontandoli con il Baseline FDCNN.

## Risultati (MAE)

| Dataset | Baseline (Senza PCA) | Con PCA (95%) | Impatto |
| :--- | :--- | :--- | :--- |
| **Albrecht** | **7.93** | 13.43 | 🔴 Peggiorato (+69%) |
| **Desharnais** | **2483** | 2927 | 🔴 Peggiorato (+18%) |
| **Maxwell** | **4642** | 6467 | 🔴 Peggiorato (+39%) |
| **COCOMO-81** | **631** | 795 | 🔴 Peggiorato (+26%) |
| **China** | **~1400** | 1635 | 🔴 Peggiorato (+16%) |

## Analisi del Fallimento
L'approccio PCA ha causato un **crollo drammatico delle prestazioni** su tutti i dataset.
Motivi:
1. **Natura Non-Supervisionata**: La PCA massimizza la varianza *indipendentemente* dal target (Effort). Nel Software Engineering, metriche con bassa varianza (es. indicatori di complessità del database) possono avere una correlazione altissima con l'effort. Scartando il 5% di varianza, la PCA ha palesemente rimosso il "segnale" vitale, lasciando solo il rumore di fondo.
2. **Distorsione Spaziale nel FCM**: Trasformare lo spazio in componenti ortogonali ha distrutto la distribuzione spaziale naturale dei progetti, portando il clustering Fuzzy C-Means a creare regioni sub-ottimali.

## Conclusione e Next Step
L'approccio 1 (PCA) è ufficialmente fallito ed è da scartare. 

Propongo di disattivare immediatamente la PCA e passare all'**Approccio 2 (Feature Selection basata su Mutual Information / Filter Method)**. A differenza della PCA, la Mutual Information è supervisionata: valuterà le singole feature in base a quanto "Effort" riescono a spiegare, scartando solo le variabili veramente inutili o puramente rumorose, mantenendo intatto lo spazio per il Fuzzy C-Means.
