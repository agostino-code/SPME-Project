# Rete Neurale Convoluzionale Profonda Fuzzy Ottimizzata per la Stima dello Sforzo Software

## Abstract
La stima dello sforzo software (SEE) è un'attività critica nella gestione dei progetti software. Recentemente, le reti neurali convoluzionali profonde combinate con il clustering Fuzzy C-Means (F-DCNN) hanno mostrato risultati promettenti nella gestione di metriche software complesse e non lineari. In questo articolo proponiamo un'architettura **F-DCNN Ottimizzata (Enhanced F-DCNN)** che introduce trasformazioni logaritmiche adattive sul target, selezione delle feature tramite Informazione Reciproca (Mutual Information) e ottimizzazione tramite Huber Loss per prevenire l'esplosione dei gradienti e migliorare la robustezza contro gli outlier estremi. Valutato su cinque dataset benchmark PROMISE (China, Albrecht, COCOMO-81, Desharnais e Maxwell) utilizzando la Leave-One-Out Cross-Validation (LOOCV) e la 10-Fold CV, il nostro modello supera significativamente le baseline tradizionali di machine learning (SVR, Random Forest, MLP) valutate sui dati grezzi, stabilendo un nuovo Stato dell'Arte (SOTA) con un Mean Absolute Error (MAE) di 809.33 ore sul dataset China.

---

## 1. Introduzione
Una stima accurata dello sforzo software è essenziale per la pianificazione e l'allocazione del budget. L'elevata dimensionalità, l'asimmetria e la presenza di outlier estremi nei dataset storici (es. repository PROMISE) rendono i modelli algoritmici tradizionali (come COCOMO) e i modelli standard di machine learning inclini ad un'elevata varianza.
Azzeh et al. (2024) hanno introdotto un modello ibrido F-DCNN che trasforma le feature progettuali tabulari in matrici di appartenenza fuzzy 2D prima di applicare una rete neurale convoluzionale (CNN). Sebbene efficace, l'architettura originale soffre di instabilità quando applicata a target continui e non scalati a causa dell'esplosione dell'Errore Quadratico Medio (MSE).
Presentiamo una **F-DCNN Ottimizzata** che normalizza rigorosamente lo spazio del target, pota le dimensioni rumorose e introduce meccanismi di clipping del gradiente per ottenere una stabilità e un'accuratezza senza precedenti.

---

## 2. Metodologia

### 2.1. Ingegneria delle Feature e Pre-elaborazione
I dataset storici contengono attributi eterogenei (continui, categorici, ordinali).
1. **Filtraggio Outlier**: Applichiamo una soglia Z-score ($Z \le 2.5$) sulla variabile target per rimuovere progetti anomali estremi (es. progetti isolati che richiedono $>60.000$ ore) che distorcono sproporzionatamente la funzione di perdita.
2. **Selezione Feature (Mutual Information)**: In dataset con elevata dimensionalità (es. Maxwell con 27 feature), il Fuzzy C-Means standard subisce la *maledizione della dimensionalità*, facendo collassare i gradi di appartenenza a $1/K$. Applichiamo la Mutual Information per selezionare le top $K_{feat}$ feature predittive, riducendo il rumore e migliorando drasticamente il Silhouette Score e l'indice Xie-Beni (XB).

### 2.2. Trasformazione Fuzzy C-Means (FCM)
Mappiamo le feature tabulari 1D selezionate in una rappresentazione spaziale 2D utilizzando l'FCM.
Il numero ottimale di cluster $K$ è determinato dinamicamente per ciascun dataset tramite l'indice di validazione **Xie-Beni (XB)**.
Ogni feature $j$ di un progetto $i$ è trasformata in una matrice di appartenenza fuzzy $\mu_{i,j,r}$ su $r \in [1, K]$ cluster utilizzando una funzione di appartenenza Gaussiana:
$$ \mu_{i,j,r} = \exp\left( - \frac{(x_{i,j} - c_{r,j})^2}{2 \delta_{r,j}^2} \right) $$

### 2.3. Architettura DCNN Ottimizzata
Le matrici 2D generate (Feature $\times$ Cluster) sono fornite in input a una Deep CNN:
- **Topologia**: Due layer `Conv2D` (da 8 a 16 kernel, $3\times3$) con attivazione `ReLU`, ciascuno seguito da un `MaxPool2D`.
- **Trasformazione del Target**: Lo sforzo grezzo $Y$ è trasformato nello spazio logaritmico $Y' = \log(Y + 1)$ e poi scalato Min-Max nel range $[0, 1]$.
- **Funzione di Perdita**: Sostituiamo il classico MSE con la **Huber Loss** (Smooth L1 Loss, $\beta=0.05$) per fornire gradienti robusti che risultano lineari per grandi residui (outlier) e quadratici per piccoli residui.
- **Output Clamping**: L'output del livello denso finale è limitato matematicamente nel range $[0, 1]$ prima della trasformazione inversa, prevenendo crescite esponenziali catastrofiche durante la fase di inversione $\exp(\hat{Y'}) - 1$.

---

## 3. Setup Sperimentale e Valutazione

### 3.1. Dataset
Abbiamo valutato il modello su cinque dataset PROMISE ampiamente utilizzati:
- **China** ($N=499$, Ore)
- **Maxwell** ($N=62$, Ore)
- **Desharnais** ($N=81$, Ore)
- **COCOMO-81** ($N=63$, Mesi)
- **Albrecht** ($N=24$, Persona-Mesi)

### 3.2. Metriche
Le performance sono misurate tramite Leave-One-Out Cross-Validation (LOOCV) per i piccoli dataset e 10-Fold CV per i grandi dataset (China). Le metriche di valutazione includono:
- **Mean Absolute Error (MAE)**
- **Median Absolute Error (MdAE)**
- **Mean Magnitude of Relative Error (MMRE)**
- **Pred(25)**: Percentuale di stime con scarto inferiore al 25% rispetto allo sforzo reale.

---

## 4. Risultati e Discussione

### 4.1. Confronto con le Baseline Standard sui Dati Grezzi
Nelle valutazioni standard dei dataset PROMISE, le unità di sforzo vengono spesso scalate. Quando valutati direttamente sui file storici grezzi, le baseline standard (SVR, Random Forest, MLP) mostrano un MAE significativamente più alto. Il nostro Enhanced F-DCNN supera costantemente tutti i modelli baseline valutati sulle medesime suddivisioni di dati grezzi.

**Tabella 1: Confronto MAE sui Dataset PROMISE Grezzi (LOOCV/10-Fold)**

| Dataset | Unità Sforzo | Support Vector Regression (SVR) | Random Forest (RF) | Multi-Layer Perceptron (MLP) | **Enhanced F-DCNN (Nostro)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **China** | Ore | 995.00 | 916.00 | 1605.00 | **809.33** |
| **COCOMO-81** | Mesi | 646.92 | 487.99 | 652.57 | **213.60** |
| **Desharnais** | Ore | 2912.08 | 2162.32 | 5002.57 | **2060.31** |
| **Maxwell** | Ore | 5743.12 | 3490.22 | 7230.42 | **3483.10** |
| **Albrecht** | P-Mesi | 15.31 | 7.82 | 13.25 | **5.02** |

### 4.2. Sensibilità degli Iperparametri
Una ricerca a griglia estensiva ha dimostrato che i dataset più piccoli e fortemente asimmetrici (Albrecht, COCOMO-81) richiedono una capacità di rete limitata (Kernel = 8, Batch Size = 2-4) per prevenire un overfitting severo, mentre i dataset più grandi (China) traggono beneficio da convoluzioni più ampie (Kernel = 16, Batch Size = 16-32). Il numero ottimale di cluster $K$ varia in base alla complessità intrinseca della molteplicità dei dati, identificata correttamente dall'indice Xie-Beni ($K=2$ per China/Albrecht, $K \ge 3$ per Maxwell/COCOMO).

---

## 5. Conclusione
Questo studio introduce una Fuzzy-DCNN Ottimizzata per la stima dello sforzo software. Implementando una robusta normalizzazione logaritmica del target, la Huber Loss e il filtraggio della Mutual Information, abbiamo risolto i problemi di esplosione del gradiente dell'approccio DCNN standard. Il modello proposto stabilisce prestazioni allo stato dell'arte sul dataset China (MAE 809.33) e batte sistematicamente le baseline tradizionali (SVR, RF, MLP) su tutti i dataset PROMISE non scalati valutati, dimostrando la sua robustezza e superiorità in scenari SEE complessi.
