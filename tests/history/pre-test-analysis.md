# Report Integrato di Analisi Pre-Test (Fasi Diagnostiche 1–7A)

## Introduzione e Scopo del Documento

Il presente documento sintetizza l'intero processo di indagine diagnostica condotto sull'architettura **Fuzzy Deep Convolutional Neural Network (F-DCNN)** per la *Software Development Effort Estimation (SDEE)*, integrando i report storici delle fasi 1–7.

L'obiettivo primario di queste fasi preliminari è stato comprendere e colmare il significativo divario di performance riscontrato nell'estrazione delle predizioni rispetto ai risultati pubblicati nello studio originale di Azzeh et al. (2024), isolando matematicamente ed empiricamente ogni componente della rete, del fuzzy clustering e della pipeline di addestramento.

---

## 1. Fase 1: Diagnostica del Baseline F-DCNN

Durante i primi test riproduttivi si è evidenziato un forte gap di performance, particolarmente pronunciato sui dataset caratterizzati da elevata asimmetria (*skewness*), come COCOMO-81 ed Albrecht. Sono stati eseguiti 8 test diagnostici sequenziali:

### 1.1 Cronologia dei Test Diagnostici
1. **Dead ReLU e Inizializzazione del Bias:**
   * *Problema:* La rete rimaneva in stallo iniziale (loss piatta, predizioni bloccate a 0).
   * *Soluzione:* Applicata l'inizializzazione Kaiming sui pesi ed inizializzato il bias del layer di output al target medio del training set, sbloccando la discesa della loss.
2. **Sensibilità agli Outlier in LOOCV:**
   * *Risultato:* Oltre il 50% del MAE totale su COCOMO-81 era generato da soli 3 fold estremi su 63, evidenziando la fragilità del MAE aggregato sotto LOOCV su campioni ridotti.
3. **Isolamento del Seed dei Worker:**
   * *Soluzione:* Fissato il seed stocastico per singolo worker/fold per evitare divergenze stocastiche.
4. **Esplosione del Gradiente su Loss MSE:**
   * *Risultato:* Su target non scalati ad alta varianza, la loss MSE generava gradienti $>10.000$, spingendo prematuramente i neuroni in saturazione ("dying ReLU").
5. **Target Scaling Lineare (Min-Max):**
   * *Risultato:* Fallimento. Comprimere il target in $[0, 1]$ non altera la distanza relativa dell'outlier. Il 70% dei fold di COCOMO continuava a collassare a 0.0.
7. **Target Scaling Logaritmico ($\log(1+y)$):**
   * *Risultato:* Risolto completamente il collasso Dead ReLU su COCOMO, ma il MAE è risultato pari a 655 (rispetto al 160 del paper) a causa di una sistematica sottostima dei valori estremi.
8. **Rappresentazione Fuzzy C-Means (FCM):**
   * Verificato che la mappatura di appartenenza fuzzy manteneva il segnale, confermando che il collo di bottiglia risiedeva nella regressione a valle e nella loss.

### 1.2 Confronto Baseline Fase 1 vs Paper Originale (Table 3, Azzeh 2024)

| Dataset | MAE Nostro | MAE Paper | Ratio (Nostro / Paper) | Stato |
| :--- | :--- | :--- | :--- | :--- |
| **China** | 1519.79 | 899 | ~1.69× | Gap contenuto su ~500 campioni |
| **Desharnais** | 2277.98 | 1044 | ~2.18× | Gap moderato |
| **Maxwell** | 6438.75 | 1749 | ~3.68× | Gap sensibile su campioni ridotti |
| **COCOMO-81** | 755.18 | 160 | ~4.72× | Gap ampio (skewness/MSE) |
| **Albrecht** | 8.65 | 1.1 | ~7.86× | Gap massimo (24 campioni) |

*Conclusione Diagnostica Fase 1:* Il divario residuo è attribuibile a scelte metodologiche non esplicitate nel testo del paper (es. pulizia pregressa degli outlier, pre-processing o trasformazioni non dichiarate).

---

## 2. Fasi 2 & 3: Evoluzione Architetturale e Spazio Latente

Dall'analisi delle limitazioni topologiche del baseline sono stati indagati la capacità della rete ed i parametri fuzzy:

1. **Capacità Neurale (Filtri Convoluzionali):**
   * L'architettura originale a 16 filtri soffriva di underfitting. La sperimentazione ha confermato che l'estensione a **32 filtri** migliora sistematicamente le prestazioni rispetto a 8 e 16 filtri.
2. **Tuning del Fuzzifier ($M$):**
   * Riducendo l'esponente di fuzzificazione $M$ da $2.0$ a **$1.5$**, l'algoritmo FCM produce appartenenze meno "sfocate" e cluster più definiti, migliorando l'estrazione di pattern convoluzionali.
3. **Cluster Fisso ($K$) vs Dinamico (XB Index):**
   * Sui **dataset grandi** (China, $>500$ righe), forzare un numero contenuto di cluster ($K=3$) previene la frammentazione eccessiva dello spazio.
   * Sui **dataset piccoli** (Desharnais, Albrecht), l'Indice Xie-Beni (XB) si è confermato superiore rispetto al $K$ fisso.

---

## 3. Fase 4: Correzione del Training, Loss Huber e Riproducibilità

In questa fase sono stati affrontati il fenomeno delle predizioni azzerate e la riproducibilità dell'ambiente.

### 3.1 Risoluzione del Collasso Dead ReLU
L'analisi dei log per-fold ha mostrato che l'attivazione `ReLU` al layer finale causava il blocco delle predizioni sul valore `target_min` (0.0):

* **Incidenza iniziale su COCOMO-81:** **67% dei fold collassati** (42/63).
* **Primo tentativo (Softplus):** Collasso ridotto al 2%, ma con introduzione di un bias positivo sistematico (`Softplus(0) ≈ 0.693`) che sovrastimava i progetti di piccole dimensioni.
* **Soluzione Definitiva:** Rete con **output lineare puro durante il training** e vincolo di non-negatività (`np.maximum(preds, 0.0)`) applicato **esclusivamente in inferenza**.

### 3.2 Transizione da RMSE a Huber Loss
Per contenere la dominanza degli outlier senza distorcere i gradienti:
* Un delta troppo grande ($\delta=0.1$ su scala $[0,1]$) equivale a MSE puro (MAE 729 su COCOMO).
* La calibrazione a **$\delta=0.01$** (equivalente alla mediana di COCOMO) ha abbattuto il MAE a **596.72** (correlazione $0.545$).

### 3.3 Problema Critico di Riproducibilità e Fix di Joblib
Durante la parallelizzazione dei fold con `joblib`, è emerso che `ThreadingBackend` condivideva lo stato RNG globale di NumPy e PyTorch tra i thread, rendendo i run non deterministici.

* **Fix:** Isolamento della logica di fold nel modulo importabile [`fold_worker.py`](../../fuzzy-DCNN-implementation/fold_worker.py) e forzatura del `backend='loky'` (multiprocesso).
* **Esito:** Ottenuto il determinismo bit-a-bit su run consecutivi.

### 3.4 Risultati e Rilevamento sulla MedAE

| Dataset | Paper MAE | Nostro MAE | Ratio MAE | Nostro MedAE | Ratio MedAE | Correlazione |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Albrecht | 1.1 | 7.10 | 6.46× | 5.10 | 4.64× | 0.92 |
| **COCOMO-81** | 160 | 676.96 | 4.23× | **111.58** | **0.70×** | 0.33 |
| Desharnais | 1044 | 2535.34 | 2.43× | 1752.77 | 1.68× | 0.52 |
| Maxwell | 1749 | 4449.28 | 2.54× | 3348.03 | 1.91× | 0.79 |
| **China** | 899 | 1398.78 | 1.56× | **688.19** | **0.77×** | 0.87 |

*Scoperta Chieve:* Su COCOMO-81 e China, la **MedAE** del modello (111.58 e 688.19) risulta **inferiore al MAE dichiarato nel paper**, fornendo un forte indizio che il paper originale possa aver riportato valori mediani o applicato un filtraggio preliminare degli outlier.

---

## 4. Fase 5: Validazione Statistica e Consolidamento

Per distinguere i reali miglioramenti dal rumore stocastico di inizializzazione dei pesi:

1. **Riproducibilità Estesa:** Confermato il determinismo bit-a-bit alla 14-esima cifra decimale su 2 run indipendenti per tutti e 5 i dataset (incluso China su GPU).
2. **Analisi Multi-Seed su COCOMO-81 (5 Seed):**
   * MAE medio: ~682 | Deviazione Standard: ~55 (~8% della media).
   * *Conclusione:* Il MAE di 596 ottenuto in Fase 4 derivava da un "lucky seed" (Seed 3: 587.91). La differenza tra delta fisso e dinamico rientra pienamente nel rumore stocastico.
3. **Sweep Delta Huber Multi-Seed (3 Seed):**
   * Averaging su più seed per moltiplicatori 0.5, 1.0 e 2.0 ha annullato le apparenze di superiorità viste nei singoli run.
4. **Verifica Architetture Alternative (`FDCNN_SeqAsym`):**
   * Su 3 seed, `FDCNN_SeqAsym` + Huber (MAE 2578) è risultata equivalente o leggermente inferiore al baseline `FDCNN` + Huber (MAE 2440), dimostrando che i presunti vantaggi erano artefatti stocastici.
5. **Indagine su Xie-Beni in COCOMO-81:**
   * Forzare $K=2$ su COCOMO ha prodotto **esattamente lo stesso output al bit** del baseline con Xie-Beni, dimostrando che l'algoritmo XB seleziona già il $K$ matematicamente ottimale ($K=2$). La bassa correlazione (0.33) è causata dall'alta dimensionalità (17 feature su 63 campioni).

---

## 5. Fasi 6 & 7A: Feature Reduction e Fallimento della PCA

Dati i limiti incontrati nel modificare l'architettura neurale su dataset così ridotti, si è ipotizzato di ridurre lo spazio delle feature per contrastare la *Maledizione della Dimensionalità*.

### 5.1 Valutazione Sperimentale della PCA (95% Varianza)

| Dataset | Baseline (Senza PCA) | Con PCA (95%) | Impatto |
| :--- | :--- | :--- | :--- |
| **Albrecht** | **7.93** | 13.43 | 🔴 Peggiorato (+69%) |
| **Desharnais** | **2483** | 2927 | 🔴 Peggiorato (+18%) |
| **Maxwell** | **4642** | 6467 | 🔴 Peggiorato (+39%) |
| **COCOMO-81** | **631** | 795 | 🔴 Peggiorato (+26%) |
| **China** | **~1400** | 1635 | 🔴 Peggiorato (+16%) |

### 5.2 Analisi delle Cause del Fallimento
1. **Natura Non-Supervisionata:** La PCA massimizza la varianza senza considerare il target (Effort). Scartando il 5% di varianza residua, ha rimosso feature chiave altamente correlate con l'effort.
2. **Distorsione dello Spazio Fuzzy:** La trasformazione in componenti ortogonali ha alterato la metrica di distanza euclidea usata da Fuzzy C-Means, degradando la qualità delle matrici di appartenenza.

*Conclusione:* L'approccio PCA è stato ufficialmente scartato in favore di tecniche di Feature Selection supervisionate (Mutual Information).

---

## 6. Quadro Riassuntivo delle Lezioni Apprese

1. **Protocollo di Validazione:** È obbligatorio mantenere LOOCV/10-fold CV priva di data leakage ed utilizzare sempre l'esecuzione multiprocesso deterministica (`backend='loky'`).
2. **Metriche d'Insieme:** MAE e MedAE devono essere valutate congiuntamente con la Standardized Accuracy (SA) e test di significatività di Wilcoxon.
3. **Rigore Statistico Multi-Seed:** Ogni variazione iperparametrica deve essere verificata su almeno 3-5 seed per escludere fluttuazioni stocastiche.
4. **Stabilità dell'Output e Loss:** La combinazione **Output Lineare + Clipping solo in inferenza + Huber Loss ($\delta \approx \text{mediana}$)** o **Target Log-Transform ($\log(1+y)$)** costituisce l'impostazione più solida per prevenire collassi ed esplosioni di gradiente.
