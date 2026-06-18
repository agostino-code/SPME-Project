# 📐 SPME — Software Project Measurement & Estimation (F-DCNN Edition)

Applicazione web per la **stima dell'effort di sviluppo software** basata sulla fusione tra **Fuzzy Logica (Fuzzy C-Means)** e **Deep Convolutional Neural Networks (DCNN)**. Il modello implementato replica le metodologie del paper *"Software effort estimation using convolutional neural network"*.

---

## 📋 Indice

- [Panoramica](#panoramica)
- [Modello Scientifico (F-DCNN)](#modello-scientifico-f-dcnn)
- [Architettura](#architettura)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Dataset Supportati](#dataset-supportati)
- [Struttura del Progetto](#struttura-del-progetto)
- [Riferimenti](#riferimenti)

---

## Panoramica

AuraEstimate (SPME) è un tool avanzato per la predizione dello sforzo nei progetti software. L'applicazione consente di caricare qualsiasi dataset di stima presente nella cartella `Software-estimation-datasets`, effettuarne un'analisi interattiva, configurare e addestrare una rete neurale DCNN preceduta da un preprocessing basato su clustering Fuzzy C-Means (FCM), visualizzare grafici di bontà di stima e calcolare stime per nuovi progetti inserendo le variabili di input.

---

## Modello Scientifico (F-DCNN)

La pipeline di addestramento e predizione segue lo schema sotto indicato:

```
[Dataset Selezionato (CSV/ARFF)]
               │
               ▼
   [Preprocessing dei Dati] (MinMax/Standard Scaler)
               │
               ▼
     [Fuzzy C-Means (FCM)]
               ├─ Identificazione dei Centroidi C_r
               └─ Calcolo larghezze d_rj per funzioni gaussiane
                       (tramite Xie-Beni Index per k ottimo)
               │
               ▼
 [Trasformazione in Matrice 2D (m × k)]
               │
               ▼
[Deep Convolutional Neural Network (DCNN)]
               ├─ Conv2D (16 filtri, kernel 3x3) + ReLU
               ├─ MaxPool2D (Adaptive, stride 1x1)
               ├─ Conv2D (16 filtri, kernel 3x3) + ReLU
               ├─ MaxPool2D (Adaptive, stride 1x1)
               ├─ Flatten
               ├─ Dense (32, ReLU) -> Dropout (0.2)
               ├─ Dense (16, ReLU)
               └─ Dense (1, Lineare)
```

### Valutazione con Cross-Validation

Per garantire l'assenza di bias:
* **Dataset piccoli (N ≤ 100)**: viene usata la Leave-One-Out Cross Validation (LOO-CV).
* **Dataset grandi (N > 100)**: viene usata la 5-Fold Cross Validation.

---

## Architettura

L'applicazione separa la logica del modello da quella web:

```
┌──────────────────┐               API Calls               ┌──────────────────┐
│                  │ ────────────────────────────────────▸ │                  │
│     Frontend     │                                       │   Flask Backend  │
│    index.html    │ ◂──────────────────────────────────── │     app.py       │
│                  │             JSON & Plot               │                  │
└──────────────────┘                                       └────────┬─────────┘
                                                                    │
                                                           ┌────────▼─────────┐
                                                           │ estimation_model │
                                                           │   .py (PyTorch)  │
                                                           └──────────────────┘
```

---

## Installazione

### Prerequisiti

* Python 3.10 o superiore
* `.venv` attivo

### Setup e Avvio

```bash
# 1. Installa le dipendenze aggiornate
pip install -r requirements.txt

# 2. Avvia l'applicazione Flask
python app.py
```

L'applicazione sarà disponibile all'indirizzo **http://localhost:5000**.

---

## Utilizzo

### Interfaccia Web

1. **Selezione Dataset**: Consente di scegliere tra diversi benchmark storici (Albrecht, China, COCOMO, Desharnais, Maxwell, NASA, ecc.).
2. **Configurazione Parametri**: Impostazione dei cluster Fuzzy ($k$ o *Auto* tramite indice Xie-Beni), Epoche, Learning Rate e Batch size.
3. **Training & Validazione**: Avvia il calcolo della cross-validation, calcola le metriche (MAE, MAR, MdAPE, $R^2$) e disegna il grafico *Actual vs Predicted*.
4. **Stima di Nuovi Progetti**: Inietta un form dinamico per inserire i valori dei parametri del progetto per calcolare lo sforzo stimato in tempo reale.

---

## Dataset Supportati

I file risiedono in `Software-estimation-datasets/` e coprono:
* **Albrecht.arff**: 24 progetti, 7 attributi funzionali.
* **China.arff**: 499 progetti, 18 attributi funzionali.
* **COCOMO-81.csv**: 63 progetti basati sul modello COCOMO.
* **Desharnais.csv**: 81 progetti.
* **Maxwell.arff**: 62 progetti, 27 attributi.
* **Nasa93.arff**: 93 progetti.
* *Altri formati CSV e ARFF* caricati e normalizzati dinamicamente.

---

## Struttura del Progetto

```
SPME-Project/
├── app.py                          # Flask app, api routes, dataset loader
├── estimation_model.py             # Implementazione Fuzzy C-Means e DCNN (PyTorch)
├── requirements.txt                # Dipendenze con PyTorch CPU
├── README.md                       # Questo file
│
├── templates/
│   └── index.html                  # Dashboard frontend (Outfit font, Glassmorphism)
│
└── Software-estimation-datasets/   # File dei dataset storici
```

---

## Riferimenti

1. **Software effort estimation using convolutional neural network** (F-DCNN Paper).
2. **Di Martino, S., Ferrucci, F., Gravino, C., et al.** (2020). *Assessing the effectiveness of approximate functional sizing approaches for effort estimation*. Information and Software Technology.
3. **Vargha, A., Delaney, H.D.** (2000). *A Critique and Improvement of the CL Common Language Effect Size Statistics*.
