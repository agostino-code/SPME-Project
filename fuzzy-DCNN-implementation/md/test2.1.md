# Report Finale: Evoluzione Architetturale e Ottimizzazione F-DCNN (Fasi 2 & 3)

Questo documento riassume le innovazioni architetturali e i risultati dell'ottimizzazione condotta nelle Fasi 2 e 3 sull'architettura originale Fuzzy Deep Convolutional Neural Network (F-DCNN), con l'obiettivo di massimizzarne le performance nella stima dell'effort software.

## 1. Debolezze del Baseline Originale (Fase 1)
L'implementazione fedele del paper originale (Azzeh et al. 2024) ha mostrato limiti intrinseci nella sua topologia:
- **Sotto-parametrizzazione**: Il limite a soli 16 filtri convoluzionali imponeva un underfitting sistematico per le feature estratte.
- **Parametri Fuzzy Standard**: Il generatore di input FCM utilizzava sempre il Fuzzifier $M=2.0$ e calcolava il numero di cluster ($K$) unicamente tramite l'Indice Xie-Beni, che non è garanzia di ottimizzazione per una loss di regressione a valle.

*Nota Capacità: La ricerca ha confermato che **32 filtri** surclassano sempre le versioni a 8 e 16 filtri, colmando il gap capacitivo.*

## 2. Ottimizzazione dello Spazio Latente (Fase 2)

Raggiunto il limite neurale, abbiamo manipolato direttamente l'input generato dall'algoritmo Fuzzy C-Means, alterando il numero di Cluster ($K$) e il livello di sfocatura ($M$).

- **Impatto del Fuzzifier**: Abbassare $M$ da $2.0$ a **$1.5$** ha portato un netto miglioramento generalizzato. Cluster più definiti e meno "sfumati" consegnano alla rete neurale una topologia di input più netta da cui estrarre pattern.
- **K Fisso vs Dinamico (XB Index)**:
  - Sui **Dataset Grandi** (come China, >500 progetti): Forzare un numero basso di cluster ($K=3$) ha evitato l'eccessiva frammentazione dello spazio e ha ottimizzato enormemente la rete.
  - Sui **Dataset Piccoli** (come Desharnais, ~80 progetti): L'Indice Xie-Beni si conferma lo strumento matematico migliore per autocalibrare i cluster rispetto alle topologie forzate.
