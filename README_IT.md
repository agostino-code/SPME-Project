# Fuzzy-DCNN: Software Development Effort & Cost Estimator

[![English Version](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📖 **Disponibile anche in lingua inglese:** Per consultare la documentazione in inglese, visualizza [README.md](README.md).

Piattaforma accademica e applicativa per la **Stima dell'Effort e dei Costi di Sviluppo Software** (*Software Development Effort Estimation - SDEE*), basata sull'architettura neurale avanzata **Fuzzy Deep Convolutional Neural Network (F-DCNN)** combinata con i modelli parametrici a Function Points (metodo Di Martino et al., 2020).

Il progetto riproduce, analizza criticamente ed estende la ricerca di **Azzeh et al. (2024)**, risolvendo le principali instabilità empiriche (neuroni morti, gradienti esplosivi e sensibilità estrema agli outlier) attraverso tecniche SOTA di regolarizzazione, funzioni di perdita robuste e trasformazione logaritmica del target.

---

## Indice dei Contenuti
- [Panoramica e Obiettivi](#panoramica-e-obiettivi)
- [Innovazioni Metodologiche Chiave](#innovazioni-metodologiche-chiave)
- [Architettura del Modello Neurale](#architettura-del-modello-neurale)
- [Applicazione Web Interattiva](#applicazione-web-interattiva)
- [Risultati Sperimentali e Benchmark](#risultati-sperimentali-e-benchmark)
- [Suite di Test e Validazione](#suite-di-test-e-validazione)
- [Struttura del Repository](#struttura-del-repository)
- [Installazione e Avvio Rapido](#installazione-e-avvio-rapido)
- [Tesi e Materiale di Presentazione](#tesi-e-materiale-di-presentazione)
- [Riferimenti Bibliografici](#riferimenti-bibliografici)
- [Licenza](#licenza)

---

## Panoramica e Obiettivi

La stima accurata dell'effort di sviluppo (espresso in **Ore-Uomo** o **Mesi-Uomo**) è uno dei problemi più critici del *Software Project Management and Economics* (SPME). I dati storici di progetti software reali presentano tipicamente:
1. **Campioni ridotti** (da 24 a 77 progetti nei dataset classici come Albrecht, COCOMO-81, Desharnais, Maxwell).
2. **Elevata asimmetria (*skewness*)** e presenza di mega-progetti outlier che alterano i gradienti e le metriche medie convenzionali.
3. **Incertezza e vaghezza intrinseca** nei requisiti funzionali nelle fasi iniziali di ciclo di vita del software.

Questo progetto risponde a tali sfide integrando:
- **Clustering Fuzzy C-Means (FCM):** Modella l'incertezza dei requisiti proiettando le metriche funzionali su funzioni di appartenenza gaussiana $\mu_{rj}(x_j)$.
- **Deep Convolutional Neural Network (DCNN):** Estrae relazioni gerarchiche cross-feature tra i gradi di appartenenza fuzzy.
- **Modelli di Regressione Lineare Semplice (SLR):** Implementazione delle formule chiuse di Di Martino et al. (2020) basate sui conteggi di Function Points (HLFPA, IFPA, FP) come termine di confronto analitico immediato.
- **Suite di Benchmark Rigorosa:** Protocollo Leave-One-Out Cross-Validation (LOOCV) e 10-Fold CV con test statistici di Wilcoxon, Standardized Accuracy (SA) ed Effect Size ($\Delta$).

---

## Innovazioni Metodologiche Chiave

Durante la fase di analisi diagnostica approfondita (documentata in [`tests/history/pre-test-analysis.md`](tests/history/pre-test-analysis.md)), sono state identificate e risolte tre vulnerabilità strutturali del modello originale di letteratura:

| Problema Identificato | Conseguenza nel Modello Originale | Soluzione Implementata nel Nostro Modello |
| :--- | :--- | :--- |
| **Dying ReLU** | L'attivazione `nn.ReLU()` in output azzerava i gradienti, causando il **collasso a 0 sul 67% dei fold di COCOMO-81**. | **Output Lineare in addestramento** con vincolo di non-negatività `np.maximum(preds, 0.0)` applicato **esclusivamente in inferenza** (`linear_clip`). Tasso neuroni morti azzerato allo 0.0%. |
| **Esplosione del Gradiente su MSE** | Loss quadratiche su target ad ampia varianza generavano gradienti $>10.000$, destabilizzando la convergenza. | Adozione sistematica della **Huber Loss (Smooth L1)** con soglia $\delta \approx 0.01$ (allineata alla mediana del target scalato). |
| **Asimmetria Estrema dei Target** | Pochi progetti fuori scala dominavano il MAE e sottostimavano i progetti ordinari. | **Target Log-Transformation ($\log(1+y)$)** durante il training e de-trasformazione esponenziale $\exp(\hat{y})-1$ in inferenza. |
| **Non-determinismo Multithreading** | I thread condividevano il generatore casuale di PyTorch/NumPy. | Isolamento dei worker di cross-validation con processo indipendente (`backend='loky'`), raggiungendo la **riproducibilità deterministica bit-a-bit**. |

---

## Architettura del Modello Neurale

L'architettura **Fuzzy-DCNN** implementata in [`fuzzy-DCNN-implementation/`](fuzzy-DCNN-implementation/) si articola in tre stadi:

```
                      [Input Progetto Software: M feature funzionali]
                                             │
                                             ▼
                             ┌──────────────────────────────┐
                             │     Fuzzy C-Means (FCM)      │
                             │  Calcolo centroidi c_rj e    │
                             │  dispersioni pesate d_rj     │
                             └──────────────────────────────┘
                                             │
                                             ▼
                      [Matrice di Appartenenza Gaussiana μ: M × K]
                                             │
                                             ▼
                             ┌──────────────────────────────┐
                             │       Primo Blocco Conv      │
                             │  Conv1D (filtri: 16 o 32)    │
                             │  BatchNorm1d + LeakyReLU     │
                             │  MaxPool1d                   │
                             ├──────────────────────────────┤
                             │      Secondo Blocco Conv     │
                             │  Conv1D + BatchNorm1d        │
                             │  LeakyReLU + MaxPool1d       │
                             ├──────────────────────────────┤
                             │     Stadio Fully Connected   │
                             │  Flatten + Linear(dim, 32)   │
                             │  Dropout (0.1 - 0.3)         │
                             │  Linear(32, 1) [Output puro] │
                             └──────────────────────────────┘
                                             │
                                             ▼
                             ┌──────────────────────────────┐
                             │     Inferenza & Clipping     │
                             │  y_pred = max(0.0, y_linear) │
                             │  De-normalizzazione Target   │
                             └──────────────────────────────┘
                                             │
                                             ▼
                                [Effort Stimato in Ore/Mesi]
```

### Formulazione Matematica delle Appartenenze
Data la feature $j$ normalizzata in $[0, 1]$ e il cluster $r \in \{1, \dots, K\}$:
$$\mu_{rj}(x_j) = \exp\left( - \frac{(x_j - c_{rj})^2}{2 \, d_{rj}^2} \right)$$
La matrice risultante $\mu \in \mathbb{R}^{M \times K}$ mappa il livello di adesione di ogni singola proprietà funzionale rispetto ai cluster storici di riferimento.

---

## Applicazione Web Interattiva

La repository include una Web Application completa e pronta all'uso sviluppata con **Flask**, che consente a project manager e ricercatori di stimare l'effort in tempo reale:

- **Input Guidato:** Inserimento delle metriche di Function Points (Input, Output, Enquiry, File, Interface).
- **Ripartizione di Manutenzione in Percentuale (%):** Configurazione percentuale opzionale per funzionalità Aggiunte (%), Modificate (%) ed Eliminate (%) rispetto alla dimensione funzionale totale (UFP), con ricalcolo real-time dei Function Point e interruttore per disattivarle.
- **Preset di Progetto:** Configurazioni rapide basate su tipologie reali di commessa (*Piccolo*, *Medio*, *Grande*, *Mega / PNRR*).
- **Profili di Produttività del Team (PDR):** Tasso di consegna stimato derivato matematicamente dai quartili della distribuzione dei 499 progetti storici del dataset China/ISBSG ($\text{PDR} = \text{Effort} / \text{AFP}$ in ore-persona/FP):
  - **4,5 h/FP ($Q_{25} = 4,3$):** 1° quartile, identifica il 25% dei progetti con la più alta velocità di consegna (team agile, elevato riuso o componenti modulari).
  - **8,1 h/FP ($Q_{50} = 8,1$):** Mediana esatta della distribuzione storica, benchmark di riferimento per lo sviluppo enterprise standard.
  - **16,0 h/FP ($Q_{75} = 15,4$):** 3° quartile, identifica progetti complessi, fortemente regolamentati o con pesante debito tecnico legacy.
  - *Personalizzato:* Inserimento manuale del PDR storico consuntivo della propria organizzazione.
- **Calcolo Economico & Mesi-Uomo:** Tariffa oraria configurabile (€/ora, default 50 €/h) con calcolo integrato di:
  - **Effort in Ore-Persona**
  - **Mesi-Uomo (PM = ore / 160)**
  - **Budget Economico Stimato (€ = ore × tariffa oraria)**
- **Stima Comparativa Simultanea:**
  - **Fuzzy-DCNN (Nostro Modello):** Stima non-lineare basata su Deep Learning e Fuzzy Clustering condizionata alla dimensione funzionale e alla produttività.
  - **M-HLFPA (Di Martino SLR):** Stima basata su High-Level Function Point Analysis ($74.9 \cdot \text{HLFPA}^{0.58}$).
  - **M-IFPA (Di Martino SLR):** Stima basata su Indicative Function Point Analysis ($539.99 \cdot \text{IFPA}^{0.272}$).
  - **M-FP (Di Martino SLR):** Stima unificata Function Points ($87.42 \cdot \text{FP}^{0.56}$).

Per avviare la web app:
```bash
python app.py
```
Accedere via browser all'indirizzo `http://127.0.0.1:5000`.

---

## Risultati Sperimentali e Benchmark

La suite di test ha confrontato il modello **Fuzzy-DCNN Ottimizzato** con i principali modelli della letteratura:
1. **Di Martino et al. (2020) (SLR):** Regressione lineare semplice sulle metriche di dimensione funzionale.
2. **Random Forest Regressor (RF):** Ensemble non-lineare ad alberi decisionali.
3. **Support Vector Regression (SVR / SVM):** Regressione a iperpiani con kernel radiale RBF.

### Tabella Comparativa sulle Metriche di Riferimento

| Dataset | Campioni | Metrica | Di Martino (SLR) | Random Forest | SVM (SVR) | **Nostro Fuzzy-DCNN** | Guadagno MedAE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **China** | 499 | **MedAE (ore)**<br>MAE (ore)<br>SA (%) | 1757.98<br>2723.13<br>44.6% | 528.10<br>1188.02<br>75.8% | 226.88<br>1324.38<br>73.1% | **117.09**<br>**817.22**<br>**83.0%** | **-77.8%** vs RF<br>**-48.4%** vs SVM |
| **COCOMO-81** | 63 | **MedAE (mesi)**<br>MAE (mesi)<br>SA (%) | 185.87<br>617.02<br>43.9% | 196.19<br>563.19<br>48.8% | 161.99<br>603.53<br>45.1% | **71.24**<br>588.67<br>**51.0%** | **-63.7%** vs RF<br>**-56.0%** vs SVM |
| **Albrecht** | 24 | **MedAE (mesi)**<br>MAE (mesi)<br>SA (%) | 7.58<br>8.75<br>64.1% | 4.29<br>7.59<br>68.9% | 4.13<br>12.62<br>48.2% | **3.11**<br>9.72<br>**76.0%** | **-27.5%** vs RF<br>**-24.7%** vs SVM |
| **Desharnais** | 77 | **MedAE (ore)**<br>MAE (ore)<br>SA (%) | 1267.56<br>2013.50<br>50.6% | 1375.61<br>2219.95<br>45.6% | 1094.91<br>2127.22<br>47.9% | **995.24**<br>2255.37<br>**50.0%** | **-27.6%** vs RF<br>**-9.1%** vs SVM |
| **Maxwell** | 62 | **MedAE (ore)**<br>MAE (ore)<br>SA (%) | 2503.58<br>3999.02<br>53.7% | 2099.73<br>3761.32<br>56.4% | **1732.94**<br>4202.70<br>51.3% | **1986.93**<br>3823.85<br>**56.0%** | **-20.6%** vs SLR |

> **Nota Metodologica:** Nei dataset SDEE la presenza di pochi progetti giganteschi distorce fortemente la media aritmetica (MAE). L'errore mediano assoluto (**MedAE**) e la **Standardized Accuracy (SA %)** costituiscono lo standard scientifico per valutare la reale capacità predittiva sulla quasi totalità dei progetti.

Tutti i dati sperimentali estesi, i test di Wilcoxon e le predizioni progetto per progetto sono analizzati nel dettaglio nel report [`tests/results/risultati_test_completi.md`](tests/results/risultati_test_completi.md).

---

## Suite di Test e Validazione

La cartella [`tests/`](tests/) racchiude una suite modulare di test scientifici automatizzati:

| Modulo | Obiettivo Sperimentale | Script | Output Principale |
| :--- | :--- | :--- | :--- |
| **Test 1: Cluster Validation** | Valutazione Silhouette, Davies-Bouldin, Calinski-Harabasz e Xie-Beni per $K \in [2, 10]$ e $M \in [1.5, 2.5]$. | [`tests/clusters/run_clusters_test.py`](tests/clusters/run_clusters_test.py) | `cluster_metrics.csv`<br>`recommendations.json`<br>`cluster_metrics_fab.csv` |
| **Test 2: Dead Neurons** | Confronto tra 5 funzioni di attivazione in output per eliminare i neuroni morti senza indurre bias positivo. | [`tests/deadneurons/run_deadneurons_test.py`](tests/deadneurons/run_deadneurons_test.py) | `activation_comparison.csv` |
| **Test 3: Grid Search** | Ricerca iperparametrica disaccoppiata a 2 stadi (Architettura $\rightarrow$ Dinamica di Addestramento). | [`tests/gridsearch/run_gridsearch_test.py`](tests/gridsearch/run_gridsearch_test.py) | `grid_results_{ds}.csv`<br>`best_hyperparameters.json` |
| **Test 4: Huber Loss** | Confronto tra RMSE, MAE e Huber Loss con diversi valori di $\delta \in [0.01, 0.5]$ sotto test di Wilcoxon. | [`tests/huber_loss/run_huber_loss_test.py`](tests/huber_loss/run_huber_loss_test.py) | `huber_results.csv`<br>`huber_results_fab.csv` |
| **Test 5: Target Transform** | Impatto della compressione logaritmica $\log(1+y)$ rispetto alla scala lineare per contenere la skewness. | [`tests/target_transform/run_target_transform_test.py`](tests/target_transform/run_target_transform_test.py) | `log_transform_comparison.csv`<br>`log_transform_comparison_fab.csv` |
| **Test 6: Baseline Comparison** | Benchmark comparativo con Di Martino (SLR), Random Forest e Support Vector Regression (SVR). | [`tests/baselines/run_baselines_test.py`](tests/baselines/run_baselines_test.py) | `baseline_comparison_results.csv` |

### Esecuzione dell'Intera Suite di Test
Per eseguire sequenzialmente tutti i 6 test con un unico comando:
```bash
python tests/run_all_tests.py
```

---

## Struttura del Repository

```
SPME-Project/
│
├── app.py                             # Server Web Flask e API REST di inferenza
├── models.py                          # Gestore dei modelli (ModelManager) e formule SLR
├── train.py                           # Pipeline di addestramento standalone e serializzazione
├── requirements.txt                   # Dipendenze Python del progetto
├── LICENSE                            # Licenza open-source MIT
├── .gitignore                         # Regole di esclusione git ottimizzate
├── .gitmodules                        # Configurazione del submodulo dataset storici
├── PLAN.md                            # Piano di sperimentazione e protocollo metodologico
├── README.md                          # Master README in lingua inglese
├── README_IT.md                       # Documentazione completa in lingua italiana
│
├── docs/                              # Documentazione e materiale illustrativo
│   └── presentazione_SPME_Fuzzy_DCNN.pptx # Slide di presentazione del progetto / tesi
│
├── fuzzy-DCNN-implementation/         # Libreria Core PyTorch + Fuzzy Logic
│   ├── DCNN.py                        # Modello neurale convoluzionale e regressore
│   ├── fuzzy.py                       # Fuzzy C-Means e membership gaussiana
│   ├── dataset.py                     # Caricamento e trasformazione dati (ARFF/CSV)
│   ├── fold_worker.py                 # Worker multiprocesso deterministico per CV
│   ├── evaluate.py                    # Calcolo metriche di valutazione
│   ├── elbow.py                       # Analisi del gomito e Xie-Beni
│   ├── corr.py                        # Matrici di correlazione
│   ├── pyproject.toml                 # Metadati del package
│   └── README.md                      # Documentazione dedicata della libreria
│
├── models/                            # Pesi pre-addestrati e metadati per l'app web
│   ├── dcnn.pth                       # Pesi salvati del modello PyTorch F-DCNN
│   ├── models-meta.joblib             # Scaler, centroidi, dispersioni e parametri di training
│   └── regression.py                  # Utility addizionali per regressione lineare
│
├── templates/                         # Template frontend
│   └── index.html                     # Interfaccia grafica utente (Glassmorphism UI)
│
├── static/                            # Asset statici
│   └── style.css                      # Foglio di stile CSS personalizzato
│
├── sample-papers/                     # Letteratura scientifica di riferimento (PDF)
│   ├── Software effort estimation using convolutional neural network.pdf  # Azzeh et al. (2024)
│   ├── Assessing the effectiveness of approximate functional sizing...    # Di Martino et al. (2020)
│   ├── SELI The Stacking-Blending Ensemble Learning... (Jayadi 2026)
│   ├── SENSE Software Effort Estimation using Novel Stacking... (2024)
│   ├── Software Cost Estimation Using TabNet and HHO (Alhumam 2025)
│   └── ...
│
├── Software-estimation-datasets/      # Submodulo Git: repository dataset SDEE storici
│   ├── china.arff                     # Dataset China (499 progetti)
│   ├── COCOMO-81.csv                  # Dataset COCOMO-81 (63 progetti)
│   ├── albrecht.arff                  # Dataset Albrecht (24 progetti)
│   ├── Desharnais.csv                 # Dataset Desharnais (77 progetti)
│   └── maxwell.arff                   # Dataset Maxwell (62 progetti)
│
├── tesi/                              # Tesi di Laurea accademica
│   ├── main.pdf                       # Documento tesi completo compilato in formato PDF
│   ├── main.tex                       # Sorgente principale LaTeX
│   └── references.bib                 # Bibliografia BibTeX
│
└── tests/                             # Suite sperimentale modulare
    ├── utils.py                       # Funzioni di supporto, metriche e LOOCV
    ├── run_all_tests.py               # Esecutore unificato di tutti i test
    ├── clusters/                      # Test 1: Validazione cluster FCM
    ├── deadneurons/                   # Test 2: Analisi neuroni morti e attivazioni
    ├── gridsearch/                    # Test 3: Grid search iperparametrica 2-stadi
    ├── huber_loss/                    # Test 4: Valutazione loss robuste
    ├── target_transform/              # Test 5: Trasformazione logaritmica target
    ├── baselines/                     # Test 6: Benchmark vs SLR, RF, SVR
    ├── history/                       # Cronologia diagnostica pre-test
    └── results/                       # Risultati consolidati e CSV per-fold
```

---

## Installazione e Avvio Rapido

### Prerequisiti
- **Python $\ge 3.12$**
- **Git**

### 1. Clonazione del Repository (con submoduli)
Poiché i dataset storici sono gestiti come submodulo Git, clonare il repository con il flag `--recursive`:
```bash
git clone --recursive https://github.com/<username>/SPME-Project.git
cd SPME-Project
```

*Se il repository è già stato clonato senza submoduli, inizializzarli con:*
```bash
git submodule update --init --recursive
```

### 2. Creazione dell'Ambiente Virtuale
```bash
# Su Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Su Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installazione delle Dipendenze
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Addestramento del Modello (Opzionale)
Il repository include già i pesi pre-addestrati in `models/dcnn.pth`. Qualora si desiderasse ri-addestrare il modello da zero sul dataset *China*:
```bash
python train.py
```

### 5. Avvio dell'Applicazione Web
```bash
python app.py
```
Aprire il browser all'indirizzo [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 6. Esecuzione dei Benchmark di Test
Per lanciare i test specifici:
```bash
# Esegue tutti i 6 test sperimentali
python tests/run_all_tests.py

# Oppure eseguire un singolo modulo:
python tests/baselines/run_baselines_test.py
python tests/clusters/run_clusters_test.py
python tests/deadneurons/run_deadneurons_test.py
python tests/huber_loss/run_huber_loss_test.py
python tests/target_transform/run_target_transform_test.py
python tests/gridsearch/run_gridsearch_test.py
```

---

## Tesi e Materiale di Presentazione

Il progetto include la documentazione accademica completa:
- **Tesi di Laurea in formato PDF:** [`tesi/main.pdf`](tesi/main.pdf)
- **Sorgente LaTeX e Bibliografia:** [`tesi/main.tex`](tesi/main.tex) e [`tesi/references.bib`](tesi/references.bib)
- **Slide di Presentazione:** [`docs/presentazione_SPME_Fuzzy_DCNN.pptx`](docs/presentazione_SPME_Fuzzy_DCNN.pptx)

---

## Riferimenti Bibliografici

1. **Azzeh, M., et al. (2024).** *Software effort estimation using convolutional neural network.* Empirical Software Engineering.
2. **Di Martino, S., Ferrucci, F., Gravino, C., & Sarro, F. (2020).** *Assessing the effectiveness of approximate functional sizing approaches for effort estimation.* IEEE Transactions on Software Engineering.
3. **Jayadi, R., et al. (2026).** *SELI: The Stacking-Blending Ensemble Learning and Interpretability Framework for Software Effort Estimation.*
4. **Kaushik, A., et al. (2024).** *SENSE: Software Effort Estimation using Novel Stacking Ensemble Learning.*
5. **Alhumam, A. (2025).** *Software Cost Estimation Using TabNet and Harris Hawks Optimization.*
6. **Hoc, N. T., et al. (2023).** *Comparing Stacking Ensemble and Deep Learning for Software Project Effort Estimation.*

---

## Autori

Sviluppato da **Agostino Cesarano** e **Fabrizio Apuzzo**.

---

## Licenza

Questo progetto è distribuito sotto licenza **MIT**. Consulta il file `LICENSE` per ulteriori informazioni. Il submodulo `Software-estimation-datasets` rispetta i termini della licenza originale dell'autore (Derek Jones).
