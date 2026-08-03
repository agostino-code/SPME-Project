# Report Diagnostico Fase 6: Validazione Architetturale Definitiva e Pulizia

In seguito alle indagini condotte in Fase 5, è emersa l'estrema importanza di valutare qualsiasi modifica architetturale o di iperparametri tramite un approccio statistico (multi-seed), a causa dell'alta sensibilità all'inizializzazione dei pesi su dataset molto piccoli.

## 1. Semplificazione del Modello: Rimozione del Multiplicatore Huber
L'esperimento multi-seed della Fase 5 ha dimostrato che modulare la loss Huber tramite un moltiplicatore dinamico (0.5 - 2.0) non portava vantaggi statisticamente significativi rispetto al semplice uso della mediana del target.
Di conseguenza:
- Tutta la logica, i parametri e le costanti relative al "multiplier" sono state rimosse dal codice.
- `evaluate.py`, `fold_worker.py` e `DCNN.py` ora utilizzano nativamente la mediana dei target di training normalizzata come `delta` per la HuberLoss in modo fisso e invisibile (senza più necessità di tuning).

## 2. Isolamento del Seed di Cross-Validation
È stato riscontrato e corretto un rischio metodologico in `evaluate.py`: il seed per l'inizializzazione della K-Fold CV (usata su China) era vincolato al seed dei pesi della rete. Questo implicava che ogni variazione del seed di addestramento alterava anche i dati di training/test, sporcando la valutazione della pura varianza di convergenza.
- **Fix**: Introdotto `CV_SPLIT_SEED = 42` fisso. Ora la divisione in fold rimane identica e puramente deterministica; il `model_seed` altera esclusivamente l'inizializzazione dei pesi.

## 3. Chiusura del Confronto Architetturale (Multi-seed)
Per verificare in modo definitivo se le "architetture esotiche" introdotte in Fase 2 (`SeqAsym`, `ParAsym`, `Pointwise`) offrissero un vero vantaggio sul baseline `FDCNN`, abbiamo eseguito una valutazione estesa con 5 seed iniziali (1, 2, 3, 4, 5) per ciascun modello.

**Risultati Aggregati (MAE Medio su 5 Seed):**

| Dataset | FDCNN (Baseline) | FDCNN_SeqAsym | FDCNN_ParAsym | FDCNN_Pointwise 1x1 |
| :--- | :--- | :--- | :--- | :--- |
| **Albrecht** | **7.93** | 9.69 | 9.24 | 14.27 |
| **Desharnais** | 2483 | 2574 | **2409** | 2811 |
| **Maxwell** | **4642** | 5804 | 4709 | 4928 |

### Analisi e Conclusione Architetturale
- **FDCNN (Baseline)** domina nettamente su Albrecht e Maxwell. Su Albrecht in particolare, qualsiasi variante asimmetrica o pointwise degrada le prestazioni in modo misurabile.
- **FDCNN_ParAsym** ottiene un lieve vantaggio matematico su Desharnais (2409 vs 2483), pari a un misero 3%. Tuttavia, osservando la deviazione standard intra-seed (es. FDCNN varia da 2298 a 2587), questo gap è ampiamente trascurabile e non giustifica la maggiore complessità topologica.
- **FDCNN_SeqAsym** e **Pointwise**, un tempo ritenuti superiori in run isolati, collassano miseramente se testati statisticamente, rivelando che i loro passati exploit erano puro overfitting o "lucky seed".

## Conclusione Finale
L'architettura baseline **FDCNN (Convoluzione Standard 2D) con Loss Huber (delta=mediana)** è l'unica configurazione robusta. 
Il filone di ricerca dedicato alla manipolazione topologica della rete (Fase 2) è ufficialmente concluso con esito negativo: aumentare o variare la complessità neurale su dataset di questo calibro amplifica solo l'instabilità stocastica senza estrarre pattern reali aggiuntivi. Ogni futuro tentativo di migliorare le prestazioni (in particolare su COCOMO-81) dovrà concentrarsi sull'input (Feature Selection, PCA, regolarizzazione dello spazio fuzzy), non sulla rete.