# Fuzzy-DCNN: Software Development Effort & Cost Estimator

[![Italian Version](https://img.shields.io/badge/Lingua-Italiano-green.svg)](README_IT.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📖 **Disponibile anche in lingua italiana:** Per consultare la documentazione completa in italiano, visualizza [README_IT.md](README_IT.md).

An academic and industrial-ready research platform for **Software Development Effort Estimation (SDEE)** combining **Fuzzy C-Means (FCM)** clustering, **Deep Convolutional Neural Networks (DCNN)**, and closed-form Function Point models based on Di Martino et al. (2020).

This project critically analyzes, reproduces, and significantly improves upon the research by **Azzeh et al. (2024)**, systematically overcoming empirical failure modes (dead neurons, exploding gradients, and high sensitivity to extreme outliers) through SOTA regularization, robust loss functions, and logarithmic target transformation.

---

## Table of Contents
- [Overview & Objectives](#overview--objectives)
- [Key Methodological Innovations](#key-methodological-innovations)
- [Neural Network Architecture](#neural-network-architecture)
- [Interactive Web Application](#interactive-web-application)
- [Experimental Results & Benchmarks](#experimental-results--benchmarks)
- [Validation & Test Suite](#validation--test-suite)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Thesis & Presentation Material](#thesis--presentation-material)
- [Scientific References](#scientific-references)
- [License](#license)

---

## Overview & Objectives

Accurate estimation of development effort (measured in **Person-Hours** or **Person-Months**) represents one of the most critical challenges in *Software Project Management and Economics* (SPME). Real-world historical software repositories typically present:
1. **Very small sample sizes** (ranging from 24 to 77 projects in classic benchmark datasets such as Albrecht, COCOMO-81, Desharnais, and Maxwell).
2. **Extreme skewness** and multi-order-of-magnitude outlier projects that corrupt conventional arithmetic mean metrics and destabilize gradient descent.
3. **Inherent uncertainty and vagueness** in early-stage software requirements.

This project addresses these challenges by integrating:
- **Fuzzy C-Means (FCM) Clustering:** Models uncertainty by mapping functional size features into Gaussian membership degrees $\mu_{rj}(x_j)$.
- **Deep Convolutional Neural Network (DCNN):** Captures high-order, cross-feature spatial relationships across fuzzy membership matrices without explosive parameter growth.
- **Parametric Function Points Models (SLR):** Closed-form equations from Di Martino et al. (2020) based on High-Level Function Points (HLFPA), Indicative Function Points (IFPA), and total Function Points (FP).
- **Statistically Rigorous Evaluation:** Full Leave-One-Out Cross-Validation (LOOCV) and 10-Fold CV with Wilcoxon signed-rank tests ($p < 0.05$), Standardized Accuracy (SA), and Effect Size ($\Delta$).

---

## Key Methodological Innovations

Through extensive diagnostic evaluations (detailed in [`tests/history/pre-test-analysis.md`](tests/history/pre-test-analysis.md)), three fundamental vulnerabilities in the literature baseline were identified and resolved:

| Identified Issue | Literature Baseline Symptom | Our Implemented SOTA Solution |
| :--- | :--- | :--- |
| **Dying ReLU Collapse** | Using `nn.ReLU()` at the output layer extinguished gradients, causing **predictions to collapse to 0.0 on 67% of COCOMO-81 folds**. | **Linear Output layer during training** with non-negative clipping `np.maximum(preds, 0.0)` applied **strictly at inference time** (`linear_clip`). Dead neuron rate reduced to **0.0%**. |
| **Exploding Gradients on MSE** | Quadratic loss on unscaled high-variance targets generated gradients exceeding $>10,000$, causing divergence. | Systematic switch to **Huber Loss (Smooth L1)** with threshold $\delta \approx 0.01$ (aligned with the median of normalized target values). |
| **Severe Target Skewness** | A few massive outlier projects inflated arithmetic MAE and led to severe underestimation of typical small-medium projects. | **Target Log-Transformation ($\log(1+y)$)** during training and exponential inverse mapping $\exp(\hat{y})-1$ during inference. |
| **Multithreading Non-determinism** | Multi-threaded execution shared the PRNG state across threads, destroying reproducibility. | Isolated process-based cross-validation workers (`backend='loky'`), achieving **exact bit-by-bit determinism**. |

---

## Neural Network Architecture

The **Fuzzy-DCNN** architecture implemented in [`fuzzy-DCNN-implementation/`](fuzzy-DCNN-implementation/) consists of three main stages:

```
                  [Software Project Functional Features: M dimensions]
                                           │
                                           ▼
                           ┌──────────────────────────────┐
                           │     Fuzzy C-Means (FCM)      │
                           │  Compute centroids c_rj and  │
                           │  weighted spreads d_rj       │
                           └──────────────────────────────┘
                                           │
                                           ▼
                    [Gaussian Membership Matrix μ: M features × K clusters]
                                           │
                                           ▼
                           ┌──────────────────────────────┐
                           │      First Conv1D Block      │
                           │  Conv1D (16 or 32 filters)   │
                           │  BatchNorm1d + LeakyReLU     │
                           │  MaxPool1d                   │
                           ├──────────────────────────────┤
                           │     Second Conv1D Block      │
                           │  Conv1D + BatchNorm1d        │
                           │  LeakyReLU + MaxPool1d       │
                           ├──────────────────────────────┤
                           │     Fully Connected Block    │
                           │  Flatten + Linear(dim, 32)   │
                           │  Dropout (0.1 - 0.3)         │
                           │  Linear(32, 1) [Pure Linear] │
                           └──────────────────────────────┘
                                           │
                                           ▼
                           ┌──────────────────────────────┐
                           │     Inference & Clipping     │
                           │  y_pred = max(0.0, y_linear) │
                           │  Target Inverse Scaling      │
                           └──────────────────────────────┘
                                           │
                                           ▼
                             [Estimated Effort (Hours/Months)]
```

### Mathematical Formulation of Fuzzy Membership
Given normalized feature $j \in [0, 1]$ and cluster $r \in \{1, \dots, K\}$:
$$\mu_{rj}(x_j) = \exp\left( - \frac{(x_j - c_{rj})^2}{2 \, d_{rj}^2} \right)$$
where $c_{rj}$ is the cluster center and $d_{rj}$ is the weighted standard deviation of feature $j$ within cluster $r$.

---

## Interactive Web Application

The project features a modern, production-grade **Flask** web application equipped with a Glassmorphism UI:

- **Guided Parameter Input:** Input fields for Function Point counters (Input, Output, Enquiry, File, Interface, Added, Changed, Deleted).
- **Realistic Project Scale Presets:** Instant loading for typical contract sizes (*Small*, *Medium*, *Large*, *Mega / PNRR*).
- **Team Productivity Profiles (PDR):** Configurable team delivery rates (High-Productivity / Agile at 4.5 h/FP, Standard Enterprise at 8.1 h/FP, Complex / Regulated at 16.0 h/FP, or custom delivery rate).
- **Economic Budgeting & Person-Months:** Configurable hourly rate (€/h, default €50/h) with automatic computation of:
  - **Effort in Person-Hours**
  - **Person-Months (PM = hours / 160)**
  - **Total Estimated Cost (€ = hours × hourly rate)**
- **Simultaneous Multi-Model Estimation:**
  - **Fuzzy-DCNN (Our Deep Learning Model):** Non-linear effort prediction dynamically conditioned on fuzzy feature topologies and team productivity.
  - **M-HLFPA (Di Martino SLR):** Closed-form High-Level Function Point estimation ($74.9 \cdot \text{HLFPA}^{0.58}$).
  - **M-IFPA (Di Martino SLR):** Indicative Function Point estimation ($539.99 \cdot \text{IFPA}^{0.272}$).
  - **M-FP (Di Martino SLR):** Unadjusted Function Point estimation ($87.42 \cdot \text{FP}^{0.56}$).

To start the application:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## Experimental Results & Benchmarks

The benchmark suite compared our **Optimized Fuzzy-DCNN** against dominant baseline models from the software engineering literature:
1. **Di Martino et al. (2020) (SLR):** Simple Linear Regression on functional size metrics.
2. **Random Forest Regressor (RF):** Decision-tree ensemble non-linear regressor.
3. **Support Vector Regression (SVR / SVM):** Hyperplane regression with radial basis function (RBF) kernel.

### Comprehensive Benchmark Table

| Dataset | Projects | Metric | Di Martino (SLR) | Random Forest | SVM (SVR) | **Our Fuzzy-DCNN** | MedAE Gain |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **China** | 499 | **MedAE (hrs)**<br>MAE (hrs)<br>SA (%) | 1757.98<br>2723.13<br>44.6% | 528.10<br>1188.02<br>75.8% | 226.88<br>1324.38<br>73.1% | **117.09**<br>**817.22**<br>**83.0%** | **-77.8%** vs RF<br>**-48.4%** vs SVM |
| **COCOMO-81** | 63 | **MedAE (mo)**<br>MAE (mo)<br>SA (%) | 185.87<br>617.02<br>43.9% | 196.19<br>563.19<br>48.8% | 161.99<br>603.53<br>45.1% | **71.24**<br>588.67<br>**51.0%** | **-63.7%** vs RF<br>**-56.0%** vs SVM |
| **Albrecht** | 24 | **MedAE (mo)**<br>MAE (mo)<br>SA (%) | 7.58<br>8.75<br>64.1% | 4.29<br>7.59<br>68.9% | 4.13<br>12.62<br>48.2% | **3.11**<br>9.72<br>**76.0%** | **-27.5%** vs RF<br>**-24.7%** vs SVM |
| **Desharnais** | 77 | **MedAE (hrs)**<br>MAE (hrs)<br>SA (%) | 1267.56<br>2013.50<br>50.6% | 1375.61<br>2219.95<br>45.6% | 1094.91<br>2127.22<br>47.9% | **995.24**<br>2255.37<br>**50.0%** | **-27.6%** vs RF<br>**-9.1%** vs SVM |
| **Maxwell** | 62 | **MedAE (hrs)**<br>MAE (hrs)<br>SA (%) | 2503.58<br>3999.02<br>53.7% | 2099.73<br>3761.32<br>56.4% | **1732.94**<br>4202.70<br>51.3% | **1986.93**<br>3823.85<br>**56.0%** | **-20.6%** vs SLR |

> **Methodological Insight:** In software effort estimation, extreme project outliers heavily distort arithmetic MAE. Median Absolute Error (**MedAE**) and **Standardized Accuracy (SA %)** represent the gold standard for assessing actual model accuracy across typical projects.

Detailed fold-by-fold predictions and Wilcoxon tests are available in [`tests/results/risultati_test_completi.md`](tests/results/risultati_test_completi.md).

---

## Validation & Test Suite

The [`tests/`](tests/) directory contains a modular, automated experimental suite:

| Test Module | Experimental Goal | Script | Key Outputs |
| :--- | :--- | :--- | :--- |
| **Test 1: Cluster Validation** | Geometric evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz, Xie-Beni) for $K \in [2, 10]$ and $M \in [1.5, 2.5]$. | [`tests/clusters/run_clusters_test.py`](tests/clusters/run_clusters_test.py) | `cluster_metrics.csv`<br>`recommendations.json`<br>`cluster_metrics_fab.csv` |
| **Test 2: Dead Neurons** | Evaluation of 5 output activation variants to eliminate zero-prediction collapse without positive bias. | [`tests/deadneurons/run_deadneurons_test.py`](tests/deadneurons/run_deadneurons_test.py) | `activation_comparison.csv` |
| **Test 3: Grid Search** | Discoupled 2-stage hyperparameter optimization (Architecture $\rightarrow$ Training Dynamics). | [`tests/gridsearch/run_gridsearch_test.py`](tests/gridsearch/run_gridsearch_test.py) | `grid_results_{ds}.csv`<br>`best_hyperparameters.json` |
| **Test 4: Huber Loss** | Comparative evaluation of RMSE, MAE, and Huber Loss ($\delta \in [0.01, 0.5]$) with Wilcoxon signed-rank tests. | [`tests/huber_loss/run_huber_loss_test.py`](tests/huber_loss/run_huber_loss_test.py) | `huber_results.csv`<br>`huber_results_fab.csv` |
| **Test 5: Target Transform** | Logarithmic transformation $\log(1+y)$ vs linear target scaling under multi-seed evaluation. | [`tests/target_transform/run_target_transform_test.py`](tests/target_transform/run_target_transform_test.py) | `log_transform_comparison.csv`<br>`log_transform_comparison_fab.csv` |
| **Test 6: Baseline Comparison** | Benchmark vs Di Martino (SLR), Random Forest, and Support Vector Regression (SVR). | [`tests/baselines/run_baselines_test.py`](tests/baselines/run_baselines_test.py) | `baseline_comparison_results.csv` |

### Running the Entire Suite
To run all 6 benchmark tests sequentially with a single command:
```bash
python tests/run_all_tests.py
```

---

## Repository Structure

```
SPME-Project/
│
├── app.py                             # Flask web server and REST API for inference
├── models.py                          # ModelManager and Di Martino closed-form SLR formulas
├── train.py                           # Standalone training pipeline and model serialization
├── requirements.txt                   # Project Python dependencies
├── LICENSE                            # MIT Open Source License
├── .gitignore                         # Optimized Git ignore patterns
├── .gitmodules                        # Dataset submodule configuration
├── PLAN.md                            # Comprehensive experimental plan & protocols
├── README.md                          # Primary English documentation
├── README_IT.md                       # Full documentation in Italian
│
├── docs/                              # Project documentation & presentation assets
│   └── presentazione_SPME_Fuzzy_DCNN.pptx # Project / thesis presentation slides
│
├── fuzzy-DCNN-implementation/         # Core PyTorch + Fuzzy Logic Library
│   ├── DCNN.py                        # Convolutional regressor model and EffortEstimator
│   ├── fuzzy.py                       # Fuzzy C-Means and Gaussian membership mapping
│   ├── dataset.py                     # ARFF/CSV dataset loader, log-scaling, splitting
│   ├── fold_worker.py                 # Deterministic multiprocess cross-validation worker
│   ├── evaluate.py                    # Evaluation metrics computation (MAE, RMSE, SA)
│   ├── elbow.py                       # Elbow method and Xie-Beni index computation
│   ├── corr.py                        # Correlation matrices and EDA helpers
│   ├── pyproject.toml                 # Package metadata
│   └── README.md                      # Dedicated package documentation
│
├── models/                            # Pre-trained model weights & inference metadata
│   ├── dcnn.pth                       # Serialized PyTorch model weights
│   ├── models-meta.joblib             # Scalers, centroids, spreads, and training hyperparameters
│   └── regression.py                  # Linear regression helper modules
│
├── templates/                         # Web frontend templates
│   └── index.html                     # Glassmorphism UI
│
├── static/                            # Static web assets
│   └── style.css                      # Custom stylesheet
│
├── sample-papers/                     # Literature reference papers (PDF)
│   ├── Software effort estimation using convolutional neural network.pdf  # Azzeh et al. (2024)
│   ├── Assessing the effectiveness of approximate functional sizing...    # Di Martino et al. (2020)
│   ├── SELI The Stacking-Blending Ensemble Learning... (Jayadi 2026)
│   ├── SENSE Software Effort Estimation using Novel Stacking... (2024)
│   ├── Software Cost Estimation Using TabNet and HHO (Alhumam 2025)
│   └── ...
│
├── Software-estimation-datasets/      # Git Submodule: historical SDEE datasets
│   ├── china.arff                     # China dataset (499 projects)
│   ├── COCOMO-81.csv                  # COCOMO-81 dataset (63 projects)
│   ├── albrecht.arff                  # Albrecht dataset (24 projects)
│   ├── Desharnais.csv                 # Desharnais dataset (77 projects)
│   └── maxwell.arff                   # Maxwell dataset (62 projects)
│
├── tesi/                              # Academic Master's Thesis
│   ├── main.pdf                       # Compiled thesis document (PDF)
│   ├── main.tex                       # Master LaTeX source
│   └── references.bib                 # BibTeX bibliography database
│
└── tests/                             # Modular experimental benchmark suite
    ├── utils.py                       # Common data loaders, metrics, and LOOCV runner
    ├── run_all_tests.py               # Unified test runner (Steps 1 to 6)
    ├── clusters/                      # Test 1: FCM cluster validation
    ├── deadneurons/                   # Test 2: Dead neuron & activation analysis
    ├── gridsearch/                    # Test 3: 2-stage grid search optimization
    ├── huber_loss/                    # Test 4: Robust loss functions evaluation
    ├── target_transform/              # Test 5: Target log-transformation analysis
    ├── baselines/                     # Test 6: Benchmark vs SLR, RF, SVR
    ├── history/                       # Pre-test diagnostic history
    └── results/                       # Consolidated benchmark results & per-fold CSVs
```

---

## Installation & Quickstart

### Prerequisites
- **Python $\ge 3.12$**
- **Git**

### 1. Clone the Repository (with Submodules)
Because historical datasets are managed via a Git submodule, clone with `--recursive`:
```bash
git clone --recursive https://github.com/<username>/SPME-Project.git
cd SPME-Project
```

*If already cloned without submodules, initialize them via:*
```bash
git submodule update --init --recursive
```

### 2. Create Virtual Environment
```bash
# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Model Training (Optional)
Pretrained weights are already supplied in `models/dcnn.pth`. To retrain from scratch on the *China* dataset:
```bash
python train.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 6. Run Benchmark Tests
```bash
# Run all 6 benchmark tests
python tests/run_all_tests.py

# Or run individual test modules:
python tests/baselines/run_baselines_test.py
python tests/clusters/run_clusters_test.py
python tests/deadneurons/run_deadneurons_test.py
python tests/huber_loss/run_huber_loss_test.py
python tests/target_transform/run_target_transform_test.py
python tests/gridsearch/run_gridsearch_test.py
```

---

## Thesis & Presentation Material

The repository contains the complete academic deliverables:
- **Master's Thesis PDF:** [`tesi/main.pdf`](tesi/main.pdf)
- **LaTeX Source & Bibliography:** [`tesi/main.tex`](tesi/main.tex) and [`tesi/references.bib`](tesi/references.bib)
- **Slide Deck:** [`docs/presentazione_SPME_Fuzzy_DCNN.pptx`](docs/presentazione_SPME_Fuzzy_DCNN.pptx)

---

## Scientific References

1. **Azzeh, M., et al. (2024).** *Software effort estimation using convolutional neural network.* Empirical Software Engineering.
2. **Di Martino, S., Ferrucci, F., Gravino, C., & Sarro, F. (2020).** *Assessing the effectiveness of approximate functional sizing approaches for effort estimation.* IEEE Transactions on Software Engineering.
3. **Jayadi, R., et al. (2026).** *SELI: The Stacking-Blending Ensemble Learning and Interpretability Framework for Software Effort Estimation.*
4. **Kaushik, A., et al. (2024).** *SENSE: Software Effort Estimation using Novel Stacking Ensemble Learning.*
5. **Alhumam, A. (2025).** *Software Cost Estimation Using TabNet and Harris Hawks Optimization.*
6. **Hoc, N. T., et al. (2023).** *Comparing Stacking Ensemble and Deep Learning for Software Project Effort Estimation.*

---

## License

This project is licensed under the **MIT License** - see the [`LICENSE`](LICENSE) file for details. The `Software-estimation-datasets` submodule is governed by its author's original licensing terms (Derek Jones).
