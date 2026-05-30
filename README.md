# 📐 SPME — Software Project Measurement & Estimation

Applicazione web per la **stima dell'effort di sviluppo software** basata su Function Point Analysis, con validazione scientifica rigorosa ispirata al metodo di *Di Martino, Ferrucci, Gravino et al. (Information and Software Technology, 2020)*.

---

## 📋 Indice

- [Panoramica](#panoramica)
- [Metodo Scientifico](#metodo-scientifico)
- [Architettura](#architettura)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Dataset](#dataset)
- [Modelli e Risultati](#modelli-e-risultati)
- [Struttura del Progetto](#struttura-del-progetto)
- [Riferimenti Bibliografici](#riferimenti-bibliografici)

---

## Panoramica

SPME è un tool di stima dei costi software che permette di:

1. **Inserire le componenti funzionali** di un progetto (EI, EO, EQ, ILF, EIF)
2. **Calcolare automaticamente** 4 metriche di dimensione funzionale (AFP, UFP, HLFPA, IFPA)
3. **Predire lo sforzo di sviluppo** in person-hours tramite modelli SLR log-log
4. **Validare scientificamente** le predizioni con test statistici (Wilcoxon, Vargha-Delaney)

Il progetto nasce nell'ambito del corso del **Dipartimento di Ingegneria dell'Informazione (DIETI)** dell'Università degli Studi di Napoli Federico II.

---

## Metodo Scientifico

L'applicazione replica la metodologia del paper:

> **"Assessing the effectiveness of approximate functional sizing approaches for effort estimation"**
> S. Di Martino, F. Ferrucci, C. Gravino et al.
> *Information and Software Technology*, 123, 106308, 2020.

### Pipeline sperimentale

```
Dataset China (499 progetti)
    │
    ├─ Calcolo metriche di size: AFP, UFP, HLFPA, IFPA
    │
    ├─ Costruzione modelli SLR in spazio log-log
    │   └─ log(EFF) = a + b × log(Size)
    │
    ├─ Validazione con Leave-One-Out Cross Validation
    │
    ├─ Valutazione accuratezza
    │   ├─ MAR  (Mean of Absolute Residuals)
    │   └─ MdAPE (Median Absolute Percentage Error)
    │
    ├─ Test statistici
    │   ├─ Wilcoxon signed-rank test (one-tailed)
    │   ├─ Correzione di Bonferroni (α/k)
    │   └─ Vargha-Delaney Â₁₂ effect size
    │
    └─ Diagnostica dei modelli
        ├─ R², Adjusted R², F-statistic, Sign.F
        ├─ Shapiro-Wilk test (normalità dei residui)
        └─ Cook's Distance (osservazioni influenti)
```

### Research Questions replicate

| RQ | Domanda | Applicabile |
|----|---------|:-----------:|
| **RQ1** | Le approssimazioni FPA superano le baselines (media/mediana)? | ✅ |
| **RQ2** | Le approssimazioni COSMIC superano le baselines? | ❌ (no dati COSMIC) |
| **RQ3** | Le approssimazioni FPA sono equivalenti all'FPA standard? | ✅ |
| **RQ4** | Le approssimazioni COSMIC sono equivalenti a COSMIC standard? | ❌ (no dati COSMIC) |

---

## Architettura

```
┌──────────────┐       POST /predict       ┌──────────────────┐
│              │ ─────────────────────────▸│                  │
│   Frontend   │       JSON response       │   Flask Backend  │
│  index.html  │ ◂─────────────────────────│     app.py       │
│              │                           │                  │
└──────────────┘                           └────────┬─────────┘
                                                    │
                                           ┌────────▼─────────┐
                                           │   china.csv      │
                                           │   499 progetti   │
                                           │   19 attributi   │
                                           └──────────────────┘
```

| Componente | Tecnologia |
|------------|-----------|
| Backend | Python 3.10+, Flask |
| Modelli | scikit-learn (`LinearRegression`), SciPy (`wilcoxon`, `shapiro`) |
| Dati | pandas, NumPy |
| Frontend | HTML5, CSS3 (glassmorphism), JavaScript vanilla |
| Font | Google Fonts (Outfit) |

---

## Installazione

### Prerequisiti

- Python 3.10 o superiore
- pip

### Setup

```bash
# 1. Clona il repository
git clone https://github.com/<your-username>/SPME-Project.git
cd SPME-Project

# 2. Crea e attiva un virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate

# 3. Installa le dipendenze
pip install -r requirements.txt
pip install scipy

# 4. Avvia l'applicazione
python app.py
```

L'app sarà disponibile su **http://localhost:5000**.

### Dipendenze

| Pacchetto | Scopo |
|-----------|-------|
| `flask` | Web server |
| `pandas` | Manipolazione dati |
| `numpy` | Calcolo numerico |
| `scikit-learn` | Modelli di regressione, LOO-CV |
| `scipy` | Test statistici (Wilcoxon, Shapiro-Wilk) |
| `matplotlib` | Visualizzazioni (script RF) |

---

## Utilizzo

### Web UI

1. Avvia il server con `python app.py`
2. Apri il browser su `http://localhost:5000`
3. Inserisci le 5 componenti funzionali del tuo progetto:
   - **EI** (External Inputs) — form, import dati
   - **EO** (External Outputs) — report, export
   - **EQ** (External Enquiries) — interrogazioni, ricerche
   - **ILF** (Internal Logical Files) — tabelle DB interne
   - **EIF** (External Interface Files) — dati esterni in sola lettura
4. Clicca **"Calcola Stima Effort"**
5. Visualizza:
   - Le 4 metriche di size (UFP, AFP, HLFPA, IFPA)
   - Le predizioni di effort per ciascun modello SLR
   - I diagnostici del modello (R², F-stat, equazione)
   - I risultati dei test di ipotesi (Wilcoxon, effect size)

### Preset disponibili

| Preset | Dimensione tipica | Componenti |
|--------|-------------------|------------|
| Piccolo (€10K) | ~19 UFP | EI=10, EO=4, EQ=3, ILF=2, EIF=0 |
| Medio (€80K) | ~148 UFP | EI=75, EO=30, EQ=25, ILF=14, EIF=4 |
| Grande (€400K) | ~735 UFP | EI=350, EO=180, EQ=120, ILF=65, EIF=20 |
| PNRR (€1.2M) | ~2200 UFP | EI=1000, EO=450, EQ=500, ILF=200, EIF=50 |

### Random Forest (script standalone)

Nella directory `random-forest-regression/` è presente uno script indipendente che addestra un modello Random Forest sullo stesso dataset:

```bash
cd random-forest-regression
python train_rf.py
```

Produce metriche (MAE, RMSE, R², MMRE, PRED(25)) e un grafico di feature importance.

---

## Dataset

### China Software Benchmark

| Proprietà | Valore |
|-----------|--------|
| **File** | `software-cost-estimation/china.csv` (e `.arff`) |
| **Provenienza** | China Software Process and Productivity Analysis Center |
| **Tipo** | Cross-company |
| **Progetti** | 499 |
| **Attributi** | 19 (tutti numerici) |
| **Valori mancanti** | Nessuno |

### Variabili principali

| Variabile | Descrizione |
|-----------|-------------|
| `AFP` | Adjusted Function Points |
| `Input` | External Inputs (EI) |
| `Output` | External Outputs (EO) |
| `Enquiry` | External Enquiries (EQ) |
| `File` | Internal Logical Files (ILF) |
| `Interface` | External Interface Files (EIF) |
| `Effort` | Effort reale in person-hours (**target**) |

### Metriche derivate (calcolate da app.py)

| Metrica | Formula | Fonte |
|---------|---------|-------|
| **UFP** | EI + EO + EQ + ILF + EIF | Standard FPA |
| **AFP** | UFP × 1.06 (per nuovi progetti) | IFPUG con VAF medio |
| **HLFPA** | 7×ILF + 5×EIF + 4×EI + 5×EO + 4×EQ | High-Level FPA (pesi complessità media) |
| **IFPA** | 35×ILF + 15×EIF | Indicative FPA (solo dati BFC) |

### Dati appalti pubblici

La directory `public-procurement/` contiene dati CSV sugli appalti pubblici IT italiani (CPV 72230000-6) dal 2020 al 2026, utili per confronti con i costi di mercato reali.

---

## Modelli e Risultati

### Risultati sul China dataset (499 progetti, LOO-CV)

| Modello | MAR | MdAPE | R² | F-stat |
|---------|-----|-------|-----|--------|
| MeanEFF (baseline) | 3700 | — | — | — |
| MedianEFF (baseline) | 3117 | — | — | — |
| **SLR AFP** | **2323** | — | 0.464 | 430.1 |
| SLR UFP | 2381 | — | 0.447 | 401.9 |
| SLR HLFPA | 2392 | — | 0.443 | 395.0 |
| SLR IFPA | 3045 | — | 0.065 | 34.4 |

### Test delle ipotesi (Wilcoxon, α Bonferroni = 0.025)

| Modello | vs MeanEFF | Effect Size | vs MedianEFF | Effect Size |
|---------|-----------|-------------|-------------|-------------|
| AFP | p < 0.001 ✅ | large | p < 0.001 ✅ | small |
| UFP | p < 0.001 ✅ | large | p < 0.001 ✅ | small |
| HLFPA | p < 0.001 ✅ | large | p < 0.001 ✅ | small |
| IFPA | p < 0.001 ✅ | medium | p < 0.001 ✅ | negligible |

### Conclusioni scientifiche

- ✅ **RQ1**: Tutti e 4 i modelli superano le baselines. AFP, UFP e HLFPA con effect size da small a large; IFPA solo con effect size negligible vs mediana.
- ✅ **RQ3**: UFP e HLFPA sono statisticamente equivalenti ad AFP (effect size negligible) → possono sostituirlo come misure rapide.
- ⚠️ **IFPA** è significativamente peggiore di AFP → sconsigliato come unico metodo di sizing.

---

## Struttura del Progetto

```
SPME-Project/
├── app.py                          # Backend Flask + modelli SLR + test statistici
├── requirements.txt                # Dipendenze Python
├── README.md                       # Questo file
│
├── templates/
│   └── index.html                  # Frontend web (glassmorphism UI)
│
├── software-cost-estimation/
│   ├── china.csv                   # Dataset China (499 progetti)
│   ├── china.arff                  # Stesso dataset in formato ARFF (Weka)
│   ├── parameters.md               # Documentazione dei parametri
│   └── china_analysis.md           # Analisi esplorativa del dataset
│
├── random-forest-regression/
│   ├── train_rf.py                 # Script Random Forest standalone
│   └── rf_visualization.png        # Grafico feature importance
│
├── public-procurement/
│   ├── 2020.csv ... 2026.csv       # Dati appalti pubblici IT
│   └── CPV 72230000-6              # Dettaglio codice CPV software
│
└── Assessing the effectiveness...pdf  # Paper di riferimento
```

---

## Riferimenti Bibliografici

1. **Di Martino, S., Ferrucci, F., Gravino, C., et al.** (2020). *Assessing the effectiveness of approximate functional sizing approaches for effort estimation*. Information and Software Technology, 123, 106308. [DOI:10.1016/j.infsof.2020.106308](https://doi.org/10.1016/j.infsof.2020.106308)

2. **Albrecht, A.J.** (1979). *Measuring application development productivity*. Proc. of the Joint SHARE, GUIDE, and IBM Application Development Symposium, 83–92.

3. **IFPUG** — International Function Point Users Group. [ifpug.org](http://www.ifpug.org)

4. **COSMIC** — Common Software Measurement International Consortium. [cosmic-sizing.org](https://cosmic-sizing.org/)

5. **Vargha, A., Delaney, H.D.** (2000). *A Critique and Improvement of the CL Common Language Effect Size Statistics of McGraw and Wong*. Journal of Educational and Behavioral Statistics, 25(2), 101–132.

6. **Wilcoxon, F.** (1945). *Individual comparisons by ranking methods*. Biometrics Bulletin, 1(6), 80–83.

---

## Licenza

Progetto accademico — Università degli Studi di Napoli Federico II, DIETI.
