# Fuzzy-DCNN Implementation

Libreria core in PyTorch e NumPy che implementa l'architettura neurale **Fuzzy Deep Convolutional Neural Network (F-DCNN)** per la stima dell'effort di sviluppo software (*Software Development Effort Estimation - SDEE*), ispirata ed estesa a partire dal lavoro di Azzeh et al. (2024).

---

## Architettura del Modello

Il modello combina la teoria della logica fuzzy con le reti neurali convoluzionali profonde per modellare l'incertezza e le relazioni non lineari tra le caratteristiche funzionali di un progetto software:

```
[Input Features: x_1, ..., x_M]
              │
              ▼
   ┌──────────────────────┐
   │    Fuzzy C-Means     │  Calcolo centroidi c_rj e dispersioni d_rj
   │  Gaussian Membership │  μ_rj = exp( - (x_j - c_rj)^2 / (2 * d_rj^2) )
   └──────────────────────┘
              │
              ▼
    [Matrice Fuzzy μ: M × K] (Trattata come immagine 1D a canali multipli)
              │
              ▼
   ┌──────────────────────┐
   │ Conv1D + BatchNorm   │  Estrazione feature locali (filtri: 16 o 32)
   │ + LeakyReLU + Pool   │
   ├──────────────────────┤
   │ Conv1D + BatchNorm   │  Estrazione pattern gerarchici cross-feature
   │ + LeakyReLU + Pool   │
   ├──────────────────────┤
   │ Fully Connected      │  Dense Layer + Dropout (0.1 - 0.3)
   ├──────────────────────┤
   │ Linear Output Layer  │  Training con output lineare non saturante
   │ (+ Clipping Infer.)  │  np.maximum(y_pred, 0.0) in inferenza
   └──────────────────────┘
              │
              ▼
    [Effort Stimato (Ore-Uomo)]
```

### 1. Fuzzificazione Gaussiana (FCM)
Dato un vettore di $M$ feature e $K$ cluster estratti tramite Fuzzy C-Means (FCM) con esponente di fuzzificazione $m$:
$$\mu_{rj}(x_j) = \exp\left( - \frac{(x_j - c_{rj})^2}{2 \, d_{rj}^2} \right)$$
dove $c_{rj}$ rappresenta il centroide del cluster $r$ per la feature $j$ e $d_{rj}$ è la deviazione standard pesata associata.

### 2. Regressione Convoluzionale Profonda (DCNN)
La matrice $\mu \in \mathbb{R}^{M \times K}$ viene processata da strati convoluzionali 1D che catturano dipendenze strutturate tra le feature funzionali senza esplosione del numero di parametri.

### 3. Soluzioni SOTA ai Problemi di Convergenza
- **Prevenzione Dying ReLU:** Strato di output lineare in addestramento con clipping non-negativo `np.maximum(y, 0.0)` in sola inferenza.
- **Huber Loss ($\delta \approx 0.01$):** Funzione di perdita robusta agli outlier estremi del software.
- **Target Log-Transform ($\log(1+y)$):** Stabilizzazione della varianza per dataset con elevata asimmetria (*skewness*).

---

## Moduli del Package

| Modulo | Descrizione |
| :--- | :--- |
| `DCNN.py` | Modello neurale PyTorch `DCNNModel` e classe wrapper ad alto livello `EffortEstimator` compatibile con API stile scikit-learn (`fit`, `predict`). |
| `fuzzy.py` | Implementazione dell'algoritmo Fuzzy C-Means `FuzzyCMeans` e trasformazione in matrici di appartenenza gaussiana. |
| `dataset.py` | Gestore dataset `Dataset`: caricamento file ARFF e CSV, normalizzazione Min-Max, log-trasformazione del target e split train/test. |
| `fold_worker.py` | Worker isolato e multiprocesso per l'esecuzione deterministica di fold di Cross-Validation (`backend='loky'`). |
| `evaluate.py` | Pipeline completa di valutazione Cross-Validation (LOOCV e 10-Fold) con calcolo delle metriche MAE, RMSE, MMRE, Pred(25) e SA. |
| `elbow.py` | Metodo del gomito e calcolo dell'indice di Xie-Beni (XB) per la selezione di $K$. |
| `corr.py` | Utilità di correlazione e analisi esplorativa tra feature. |

---

## Installazione e Uso

### Requisiti
- Python $\ge 3.12$
- PyTorch $\ge 2.2$
- NumPy, Pandas, Scikit-Learn, Scikit-Fuzzy

### Esempio Rapido di Utilizzo

```python
from dataset import Dataset
from fuzzy import FuzzyCMeans
from DCNN import EffortEstimator

# 1. Carica il dataset
ds = Dataset("Software-estimation-datasets/china.arff", target_column_name="Effort", use_log_target=True)

# 2. Addestra il clustering fuzzy
fcm = FuzzyCMeans(ds, num_clusters=5, m=2.0)
mu_train = fcm.fit_transform()

# 3. Inizializza e addestra il regressore DCNN
estimator = EffortEstimator(
    m_features=mu_train.shape[1],
    k_clusters=5,
    lr=1e-3,
    batch_size=16,
    epochs=200,
    loss_type="huber",
    huber_delta=0.01,
    activation_type="linear_clip"
)
estimator.fit(mu_train, ds.training_target_scaled.values)

# 4. Predizione su nuovi campioni
mu_test = fcm.transform(ds.testing_data)
pred_scaled = estimator.predict(mu_test)
pred_effort = ds.inverse_transform_target(pred_scaled)
```
