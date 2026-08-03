# Proposta Fase 4: Feature Reduction e Pulizia dello Spazio

Come abbiamo concluso nella Fase 3, manipolare l'architettura della rete produce solo overfitting. Il vero collo di bottiglia per le prestazioni (e la stabilità) su dataset così piccoli (Albrecht N=24, Desharnais N=81) è la **Maledizione della Dimensionalità**. 

Il clustering Fuzzy C-Means (usato per generare le membership che alimentano la rete) fatica enormemente negli spazi ad alta dimensionalità perché le distanze euclidee perdono di significato. Ridurre e pulire lo spazio delle feature prima della fuzzificazione è il passo successivo più logico.

Ecco tre approcci proposti per la Fase 4:

## 1. Principal Component Analysis (PCA)
Molte metriche del software (es. Function Points, Lines of Code, Team Size) sono fortemente correlate tra loro (multicollinearità).
- **Come funziona**: Applichiamo PCA sui dati di training (dopo lo scaling) per estrarre le componenti principali che spiegano il 90-95% della varianza. 
- **Integrazione**: Invece di passare `m_features` originali al Fuzzy C-Means, passiamo le `N` componenti principali. La DCNN avrà un tensore di input molto più compatto (`N x k_clusters`), riducendo i pesi e il rischio di memorizzazione del rumore.
- **Vantaggi**: Ortogonalizza lo spazio, migliora nettamente la qualità dei cluster fuzzy, riduce drasticamente i parametri della rete.
- **Svantaggi**: Si perde l'interpretabilità diretta delle singole feature (ma per una rete neurale "black-box" non è un problema).

## 2. Feature Selection basata su Mutual Information (MI)
Selezionare solo il sottoinsieme di feature originali che contengono il maggior "segnale" rispetto al target (Effort).
- **Come funziona**: Calcoliamo la Mutual Information tra ogni feature e il target sul training set. Selezioniamo la Top-K (o le feature sopra una certa soglia).
- **Integrazione**: Il taglio viene fatto in `dataset.py` o all'inizio di `fold_worker.py` (rigorosamente solo sul training set per evitare leakage).
- **Vantaggi**: Mantiene l'interpretabilità delle feature. Elimina variabili puramente rumorose.

## 3. Regularized Feature Elimination (RFE / Lasso)
Usare un modello lineare robusto (es. Lasso/L1 o Ridge) per identificare quali feature hanno un impatto nullo o quasi nullo sull'effort, scartandole.
- **Vantaggi**: Tiene conto delle interazioni lineari tra feature meglio della semplice Mutual Information.
