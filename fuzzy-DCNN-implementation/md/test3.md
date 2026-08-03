# Report Diagnostico Fase 5: Validazione e Consolidamento F-DCNN

Questo documento riporta i risultati della Fase 5, dedicata alla validazione rigorosa del modello F-DCNN, all'analisi di significatività statistica, e all'espansione dei test di riproducibilità identificati come necessari al termine della Fase 4 (`test2.md`).

## 1. Task 3: Riproducibilità Estesa
Il fix per il determinismo (`fold_worker.py` + backend `loky` in joblib) era stato testato solo su COCOMO-81. Abbiamo esteso la verifica a **tutti gli altri dataset**.
- **Risultato**: Eseguendo due run indipendenti per ciascun dataset (Albrecht, Desharnais, Maxwell, China) i risultati (MAE e MedAE) sono risultati **bit-identici** alla 14esima cifra decimale. 
- **China (GPU + n_jobs=1)**: Anche il dataset China, eseguito in modo sequenziale su GPU, ha mantenuto il determinismo. 
- **Conclusione**: La pipeline di valutazione è ora matematicamente e computazionalmente deterministica su tutti gli scenari.

## 2. Task 1: Confronto delta fisso vs dinamico su COCOMO-81
Nella Fase 4, la loss Huber fissa (`0.01`) sembrava aver battuto il delta dinamico (`mult=1.0`) (597 vs 677). Abbiamo ri-eseguito un test rigoroso:
- **Dinamico (`mult=1.0`)**: MAE 676.96 | MedAE 111.58
- **Fisso (`0.01`)**: MAE 631.70 | MedAE 125.91
Il gap si è ristretto, ma permane una modesta differenza a favore del fisso sul MAE. Per capire se tale differenza è reale, vedi il **Task 5**.

## 3. Task 5: Analisi di Significatività Statistica
Per misurare l'impatto dell'inizializzazione dei pesi della rete neurale sull'oscillazione dei risultati (varianza stocastica intra-modello), abbiamo testato 5 seed (1, 2, 3, 4, 5) per il training su COCOMO-81 con delta fisso = 0.01:
| Seed | MAE | MedAE |
| :--- | :--- | :--- |
| 1 | 761.12 | 149.21 |
| 2 | 697.36 | 152.99 |
| 3 | **587.91** | 103.66 |
| 4 | 691.24 | 174.36 |
| 5 | 674.08 | 116.10 |

- **Media MAE**: ~682
- **Deviazione Standard MAE**: ~55 (circa 8% della media).
- **Conclusione**: La differenza di 45 punti di MAE tra configurazione fissa (632) e dinamica (677) è **pienamente all'interno del rumore stocastico** di inizializzazione. Il MAE di 596.72 ottenuto in Fase 4 non derivava dalla superiorità della metrica fissa, ma da un "lucky seed" (come il Seed 3 qui sopra che tocca 587.91). Pertanto, la ricalibrazione dinamica non è inferiore a quella fissa in senso statisticamente significativo.

## 4. Task 2: Sweep di calibrazione del delta dinamico (Multi-seed)
Accettando la lezione del Task 5, abbiamo rieseguito lo sweep dei moltiplicatori (0.5, 1.0, 2.0) su tutti gli altri dataset usando **3 seed diversi**, per calcolare il MAE medio e filtrare il rumore:

| Dataset | Media MAE (mult=0.5) | Media MAE (mult=1.0) | Media MAE (mult=2.0) |
| :--- | :--- | :--- | :--- |
| Albrecht | **7.36** | 7.79 | 7.57 |
| Desharnais | 2478 | **2440** | 2472 |
| Maxwell | **4504** | 4548 | 4506 |
| China | **1297** | 1401 | 1398 |

- **Conclusione**: Le enormi differenze viste nel singolo run (es. 300 punti su Maxwell o Desharnais) sono **evaporate** mediando su soli 3 seed. La variazione del moltiplicatore Huber tra 0.5 e 2.0 non produce deviazioni statisticamente significative dal baseline. L'unica debole tendenza è un leggero vantaggio per `0.5` su China e Albrecht, ma ampiamente dentro il margine di rumore.

## 5. Task 6: Verifica architetture alternative (Multi-seed)
Abbiamo ri-testato la variante `FDCNN_SeqAsym` incrociata con la loss Huber su Desharnais, stavolta confrontandola con il Baseline `FDCNN` su **3 seed condivisi**:
- `FDCNN` Baseline + Huber: MAE medio **2440** (Seed 1: 2555, Seed 2: 2215, Seed 3: 2551)
- `FDCNN_SeqAsym` + Huber: MAE medio 2578 (Seed 1: 2556, Seed 2: 2567, Seed 3: 2611)
- **Conclusione**: Il "miglioramento" di 300 punti osservato nel singolo run (2231 vs 2535) era una **pura illusione stocastica** ("lucky seed"). Valutata su più inizializzazioni, la variante asimmetrica sequenziale in realtà performa leggermente *peggio* del baseline con loss Huber. La superiorità asimmetrica non regge al test di riproducibilità statistica.

## 6. Task 4: Indagine sulla bassa correlazione di COCOMO-81
Sulla bassa correlazione di COCOMO (0.33 contro i 0.8-0.9 degli altri), abbiamo indagato:
1. **Rumore Feature**: L'analisi di varianza e correlazione feature-feature ha mostrato distribuzioni uniformi e assenza di multi-collinearità estrema.
2. **Numero di cluster (K) fisso vs Xie-Beni**: Forzare K=3 o K=4 peggiora drasticamente MAE e correlazione. Forzare K=2 produce **esattamente lo stesso output al bit** (MAE 676.9604759101269) del baseline dinamico. Questo dimostra matematicamente (non per inferenza) che l'indice Xie-Beni sceglie già autonomamente l'ottimo K=2. Il problema non è l'algoritmo di selezione.
3. **Dimensionalità**: Avendo 17 feature su soli 63 campioni, e con K=2 (solo 2 centroidi per rappresentare la complessità), lo spazio fuzzy si appiattisce. Necessaria in futuro una riduzione tramite PCA o Feature Selection.

## 7. Task 7: Rischio contaminazione Fasi 1-3
La volatilità dimostrata dal multi-seed conferma che i numeri puntuali delle Fasi 1-3 (calcolati pre-fix determinismo) sono altamente inaffidabili. Qualsiasi "miglioramento" sotto il 10% riportato in passato deve essere retroattivamente declassato a rumore stocastico fino a prova contraria multi-seed.

## Conclusione
Questa fase di validazione ha operato una "distruzione creatrice" sui risultati precedenti. Abbiamo stabilito che la pipeline è ora perfettamente deterministica. Tuttavia, applicando il necessario rigore statistico (multi-seed), gran parte delle "scoperte" su iperparametri fini (delta multiplier) e architetture esotiche (SeqAsym) si sono rivelate illusioni dovute alla fragilità numerica di dataset minuscoli (come Desharnais o Maxwell). 
Il vero collo di bottiglia non risiede nei micro-tuning dell'architettura DCNN o della loss, ma nell'incapacità matematica di estrarre segnale stabile da spazi input a 17 dimensioni con soli 60-80 sample (curse of dimensionality). Il prossimo passo obbligato è spostare l'attenzione dall'architettura neurale alla **riduzione e pulizia dello spazio delle feature** pre-fuzzificazione.
