# Report Finale: Evoluzione Architetturale e Ottimizzazione F-DCNN (Fasi 2 & 3)

Questo documento riassume le innovazioni architetturali e i risultati dell'ottimizzazione condotta nelle Fasi 2 e 3 sull'architettura originale Fuzzy Deep Convolutional Neural Network (F-DCNN), con l'obiettivo di massimizzarne le performance nella stima dell'effort software.

## 1. Debolezze del Baseline Originale (Fase 1)
L'implementazione fedele del paper originale (Azzeh et al. 2024) ha mostrato limiti intrinseci nella sua topologia:
- **Spazialità Inesistente**: Trattare la matrice di input (feature $\times$ cluster) tramite classici kernel quadrati $3 \times 3$ forzava la rete a cercare correlazioni locali tra metriche software logicamente indipendenti (es. `Lines of Code` e `Language`), a differenza dei pixel naturali in un'immagine.
- **Sotto-parametrizzazione**: Il limite a soli 16 filtri convoluzionali imponeva un underfitting sistematico per le feature estratte.
- **Parametri Fuzzy Standard**: Il generatore di input FCM utilizzava sempre il Fuzzifier $M=2.0$ e calcolava il numero di cluster ($K$) unicamente tramite l'Indice Xie-Beni, che non è garanzia di ottimizzazione per una loss di regressione a valle.

---

## 2. Esplorazione Architetturale (Fase 2)

Per risolvere il problema della spazialità, abbiamo ideato due innovazioni e condotto una Grid Search (5-fold CV):

### A. Fusione Canali via Convoluzione Pointwise ($1 \times 1$)
Al modello base (`FDCNN`) è stata aggiunta una convoluzione $1 \times 1$ con `BatchNorm2d` interposta. Questo ha permesso di espandere e mescolare i canali latenti (fusione di densità) prima del pooling, senza ampliare il campo ricettivo spaziale su metriche non correlate.
* **Risultato**: Ha stabilito il nuovo record sul dataset **China** (MAE crollato da 1519 a **1459**).

### B. Kernel Asimmetrici Disaccoppiati (Sequenziali)
Abbiamo creato la variante `FDCNN_SeqAsym`, sostituendo il kernel $3 \times 3$ con due step successivi:
1. Convoluzione $(1 \times K)$ per comprimere le probabilità cluster lungo una singola feature.
2. Convoluzione $(M \times 1)$ per aggregare i risultati tra le diverse feature.
* **Risultato**: Ha dominato sul dataset **Desharnais** (MAE ridotto da 2277 a **2040**). I kernel asimmetrici paralleli hanno invece sofferto di troppa frammentazione del segnale e sono stati scartati.

*Nota Capacità: La ricerca ha confermato che **32 filtri** surclassano sempre le versioni a 8 e 16 filtri, colmando il gap capacitivo.*

---

## 3. Ottimizzazione dello Spazio Latente (Fase 3)

Raggiunto il limite neurale, abbiamo manipolato direttamente l'input generato dall'algoritmo Fuzzy C-Means, alterando il numero di Cluster ($K$) e il livello di sfocatura ($M$).

- **Impatto del Fuzzifier**: Abbassare $M$ da $2.0$ a **$1.5$** ha portato un netto miglioramento generalizzato. Cluster più definiti e meno "sfumati" consegnano alla rete neurale una topologia di input più netta da cui estrarre pattern.
- **K Fisso vs Dinamico (XB Index)**:
  - Sui **Dataset Grandi** (come China, >500 progetti): Forzare un numero basso di cluster ($K=3$) ha evitato l'eccessiva frammentazione dello spazio e ha ottimizzato enormemente la rete, portando il MAE a **1271**.
  - Sui **Dataset Piccoli** (come Desharnais, ~80 progetti): L'Indice Xie-Beni si conferma lo strumento matematico migliore per autocalibrare i cluster rispetto alle topologie forzate.

---

## 4. Configurazioni Vincenti Definitive

| Dataset (Size) | Modello Consigliato | Filtri / Dropout | Parametri FCM | MAE Ottimizzato | MAE Baseline (Fase 1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **China** (Grande) | `FDCNN` + Pointwise 1x1 | 32 / 0.2 | $K=3$, $M=1.5$ | **1271.49** | 1519.79 |
| **Desharnais** (Piccolo)| `FDCNN_SeqAsym` | 32 / 0.5 | XB Index, $M=2.0$ | **2040.56** | 2277.98 |

## Conclusione

Trattare l'architettura per la stima dell'effort non come un "black box" standard per la Computer Vision, ma applicando convoluzioni mirate (Pointwise e Kernel Asimmetrici 1D sequenziali) ha dimostrato di ridurre enormemente l'errore assoluto medio. Allo stesso tempo, si è confermato che la qualità della regolarizzazione fuzzy d'ingresso (basso fuzzifier, controllo della frammentazione) è determinante almeno quanto i pesi della rete stessa.
