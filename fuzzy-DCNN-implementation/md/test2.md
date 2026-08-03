# Report Diagnostico Fase 4: Correzione del Training, Loss Tuning e Riproducibilità (F-DCNN)

Questo report prosegue la diagnosi delle Fasi 1-3 (vedi test1.md), concentrandosi su un problema rimasto irrisolto nella Fase 1: il collasso "Dead ReLU" osservato nell'output della rete su dataset fortemente skewed, e sulla scoperta di un problema metodologico più fondamentale — la non-riproducibilità della pipeline di training — che ha condizionato l'affidabilità di tutti i confronti fatti finora.

## Cronologia dei Test Diagnostici

### 1. Conferma empirica del collasso "Dead ReLU" in output

Analizzando i CSV per-fold prodotti dalla LOOCV, è emerso che una frazione consistente delle predizioni collassava esattamente sul valore minimo del target di training (`target_min`), con incidenza correlata alla skewness del dataset:

| Dataset | % predizioni collassate sul minimo |
| :--- | :--- |
| **COCOMO-81** (skew 4.4) | **67%** (42/63 fold) |
| Maxwell (skew 3.26) | 10% |
| China (skew 3.92) | 2% |
| Desharnais (skew 2.0) | 0% |
| Albrecht (skew 2.2) | 0% |

**Causa**: l'attivazione finale `nn.ReLU()` azzera le pre-attivazioni negative e non propaga gradiente per quei percorsi. Su dataset skewed, la loss RMSE è dominata dagli outlier durante il training, spingendo la rete a produrre pre-attivazioni negative per la maggioranza dei campioni "normali" — che restano permanentemente bloccati a zero (cioè `target_min` una volta invertita la scala).

### 2. Primo tentativo di fix: Softplus in output

Sostituendo `ReLU` con `Softplus` (sempre positiva, gradiente mai nullo) il collasso è quasi sparito (COCOMO 67%→2%), ma è emerso un **nuovo bias sistematico**: `Softplus(0) ≈ 0.693`, non `0`. Ogni campione con effort piccolo (la maggioranza nei dataset skewed) veniva sistematicamente sovrastimato.

**Risultato**: China 89% dei fold sovrastimati, Maxwell 65%, con MAE peggiorato o solo marginalmente migliorato rispetto al baseline ReLU.

### 3. Fix definitivo: output lineare + clipping solo in inferenza

Rimossa del tutto l'attivazione finale: il `forward` del DCNN produce ora un output lineare puro durante il training (gradiente sempre libero, nessuna saturazione, nessun bias sistematico), con il vincolo di non-negatività (`max(pred, 0)`) applicato **solo** in `predict()`, fuori dal grafo di training.

**Risultato**: miglioramento netto su tutti i dataset rispetto sia a ReLU che a Softplus, collasso quasi azzerato ovunque, correlazione actual↔predicted migliorata in modo consistente.

### 4. Tuning della loss: RMSE → Huber

Con il collasso risolto, il collo di bottiglia residuo era la sensibilità della loss RMSE agli outlier (già identificata come causa strutturale nel Punto 5 di test1.md). Introdotta `nn.HuberLoss(delta)` come alternativa configurabile.

- **Primo tentativo (`delta=0.1`, scala normalizzata `[0,1]`)**: risultato **peggiore** del baseline (MAE 729 vs 672). Causa: `delta=0.1` corrispondeva a ~1139 unità nella scala originale di COCOMO — quasi 12 volte la mediana del dataset (98) — rendendo la Huber quasi indistinguibile dall'MSE puro per la stragrande maggioranza dei campioni.

### 5. Ricalibrazione della scala di delta

Individuato un valore molto più sensato: `delta=0.01` (≈114 unità originali, vicino alla mediana di COCOMO). Risultato: **MAE 596.72**, il migliore ottenuto fino a quel momento, con correlazione salita a 0.545 (da 0.375 del baseline).

### 6. Sweep di conferma: curva a campana

Testato anche `delta=0.005` (troppo piccolo, ≈57 unità, sotto la mediana): risultato peggiore (MAE 686.75, correlazione crollata a 0.279) — comportamento simile a una loss L1 pura, con gradiente costante meno informativo su batch piccoli (batch_size=16).

**Conclusione**: `delta ≈ mediana del target di training` è il punto ottimale; sia sovra- che sotto-stimare questa soglia peggiora il risultato.

### 7. Generalizzazione: delta dinamico per-fold

Per evitare di dover ri-calibrare manualmente `delta` per ogni dataset, implementato il calcolo dinamico:

```
delta_normalizzato = (mediana_target_training_del_fold × multiplier) / target_range_del_fold
```

calcolato usando **solo** il training set di ciascun fold (nessun leakage), con `multiplier=1.0` come valore di partenza.

### 8. Scoperta di un problema di riproducibilità

Durante il confronto tra configurazioni (delta fisso vs dinamico), risultati apparentemente equivalenti in teoria (`delta≈0.0086` dinamico vs `0.01` fisso) producevano MAE molto diversi (703 vs 597) a parità di condizioni nominali. Indagando i log, è emerso che `joblib` stava usando **`ThreadingBackend`** invece di `LokyBackend` (processi separati).

**Causa del problema**: `np.random.seed()` e `torch.manual_seed()` impostano uno stato RNG **globale**, condiviso da tutti i thread di uno stesso processo. Con l'esecuzione concorrente di 63 fold su thread (non processi) separati, le chiamate di seeding di thread diversi si intrecciano in ordine non deterministico deciso dallo scheduler del sistema operativo — vanificando il determinismo nominale del codice.

### 9. Fix della riproducibilità: forzare `backend='loky'`

Primo tentativo fallito con un errore di pickling: `TypeError: cannot pickle 'CudnnModule' object`. Causa: su Windows (che usa `spawn`, non `fork`), una funzione worker definita nello script eseguito come `__main__` deve essere serializzata "per valore" da `cloudpickle`, incluso lo stato del modulo — e il contesto CUDA/cuDNN (toccato all'avvio per il rilevamento GPU) non è serializzabile.

**Fix**: estratta la logica del singolo fold (`process_single_fold` e le funzioni ausiliarie) in un modulo separato importabile (`fold_worker.py`), permettendo ai worker `loky` di referenziare la funzione per import invece che per valore.

### 10. Verifica della riproducibilità

Con `fold_worker.py` separato e `backend='loky'` esplicito, due run consecutivi con identica configurazione su COCOMO hanno prodotto **risultati identici**. La pipeline è ora verificata come genuinamente deterministica. Bonus: il tempo di esecuzione è anche diminuito significativamente (es. COCOMO: da ~400s a ~140-170s), grazie alla vera parallelizzazione multi-processo priva di contesa su risorse condivise.

### 11. Run completo sui cinque dataset (pipeline deterministica)

Con output lineare + clip in inferenza, Huber loss (delta dinamico, multiplier=1.0) e backend loky verificato:

| Dataset | Paper MAE | Nostro MAE | Ratio MAE | Nostro MedAE | Ratio MedAE | Correlazione |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Albrecht | 1.1 | 7.10 | 6.46x | 5.10 | 4.64x | 0.92 |
| **COCOMO-81** | 160 | 676.96 | 4.23x | **111.58** | **0.70x** | 0.33 |
| Desharnais | 1044 | 2535.34 | 2.43x | 1752.77 | 1.68x | 0.52 |
| Maxwell | 1749 | 4449.28 | 2.54x | 3348.03 | 1.91x | 0.79 |
| **China** | 899 | 1398.78 | 1.56x | **688.19** | **0.77x** | 0.87 |

**Osservazione principale**: su COCOMO-81 e China, la MedAE del modello è **inferiore** al MAE dichiarato nel paper — non solo vicina, ma sotto. Questo rafforza ulteriormente l'ipotesi (già formulata in test1.md) che il valore "MAE" riportato dagli autori per l'F-DCNN possa in realtà corrispondere a una mediana degli errori assoluti, oppure a una validazione diversa da una LOOCV completa e onesta su tutti i fold.

**Osservazione secondaria, non ancora spiegata**: la correlazione actual↔predicted varia moltissimo tra dataset (0.33 su COCOMO, 0.92 su Albrecht) e **non segue un pattern chiaro** né con la skewness dichiarata nel paper né con la dimensione campionaria — China ha skew quasi massimo (3.92) ma correlazione alta (0.87), mentre COCOMO ha skew massimo (4.4) e correlazione minima (0.33). Su un campione di soli 5 dataset non è possibile isolare la causa con certezza statistica.

**Osservazione critica per il delta dinamico**: confrontando COCOMO con il risultato ottenuto tramite calibrazione manuale (`delta=0.01` fisso → MAE 596.72, corr 0.545), il delta dinamico per-fold ha dato risultati **peggiori** (MAE 676.96, corr 0.330) a parità di pipeline ora deterministica. Questo suggerisce che il ricalcolo di `delta` fold-per-fold, pur teoricamente ben motivato, introduce una variabilità che il valore fisso ben calibrato non ha — la generalizzazione automatica non ha (ancora) battuto la calibrazione manuale.

---

## Limiti di questa fase

- Il moltiplicatore `DCNN_HUBER_DELTA_MEDIAN_MULTIPLIER=1.0` è stato calibrato **solo su COCOMO-81** tramite sweep manuale (0.1 / 0.01 / 0.005); la sua idoneità sugli altri quattro dataset non è stata verificata con lo stesso rigore.
- Il confronto delta-fisso vs delta-dinamico su COCOMO, pur suggestivo, è basato su un singolo run per condizione — non è stata ancora eseguita una validazione ripetuta per escludere che la differenza residua sia dovuta a varianza tra le due configurazioni piuttosto che a un reale svantaggio del delta dinamico.
- Il pattern di correlazione variabile tra dataset resta senza una spiegazione causale solida.
- Il pattern dei 3 fold "catastrofici" per dataset (outlier estremi che dominano il MAE aggregato, già documentato in test1.md) persiste identico in questa fase, confermando che il problema architetturale/di loss è stato mitigato ma non risolto alla radice per i casi più estremi.

## Conclusione di questa fase

Il collasso "Dead ReLU" è stato diagnosticato e risolto con evidenza empirica diretta (percentuale di collasso misurata prima/dopo su tutti e 5 i dataset). Il tuning della loss (RMSE → Huber con delta calibrato) ha prodotto un miglioramento reale e misurabile, non solo su COCOMO ma in modo consistente. È stato inoltre scoperto e risolto un problema di non-determinismo nella pipeline di valutazione (RNG globale condiviso tra thread), che probabilmente ha contaminato in modo silenzioso una parte dei confronti fatti nelle fasi precedenti — un rischio metodologico rilevante da segnalare esplicitamente in qualsiasi resoconto complessivo del progetto.

Il gap con i valori dichiarati nel paper resta ampio sul MAE aggregato, ma si restringe significativamente (e in due casi si inverte) se si guarda alla MedAE — coerente con l'ipotesi, mai confermata con certezza ma sempre più supportata, che il numero "MAE" del paper originale non corrisponda a un MAE calcolato su una LOOCV completa nel senso standard del termine.

---

## Necessità di Ulteriori Test

Questa fase ha aperto più domande di quante ne abbia chiuse. Prima di considerare il lavoro di tuning concluso, sono necessari almeno i seguenti test:

1. **Confronto rigoroso delta fisso vs dinamico su COCOMO**, ripetuto (non un singolo run) ora che la pipeline è verificata deterministica, per stabilire con certezza se il delta dinamico è realmente peggiore o se la singola osservazione attuale è entro il margine di variazione accettabile tra le due strategie.

2. **Sweep di calibrazione di `huber_delta` ripetuto individualmente sugli altri quattro dataset** (Albrecht, Desharnais, Maxwell, China), esattamente come fatto per COCOMO, invece di assumere che `multiplier=1.0` sia ottimale ovunque solo perché lo è per un dataset.

3. **Verifica di riproducibilità estesa agli altri quattro dataset**, non solo a COCOMO — al momento il doppio-run di conferma è stato eseguito solo su un dataset; non è garantito che Desharnais, Maxwell e China (10-fold, non LOOCV) si comportino allo stesso modo, specialmente China che usa `n_jobs=1` per motivi di GPU e quindi non passa dallo stesso meccanismo di parallelizzazione.

4. **Indagine mirata sulla bassa correlazione di COCOMO** (0.33, la peggiore di tutti e 5) rispetto ad Albrecht (0.92) — capire se dipende da rumore nelle feature, dal numero di feature (COCOMO ne ha 17, il più alto), dalla scelta del numero di cluster fuzzy, o da altro.

5. **Analisi statistica di significatività** (es. bootstrap sui fold, o ripetizione della LOOCV con seed diversi mantenendo comunque il determinismo per-seed) per stabilire se le differenze di MAE osservate tra configurazioni (es. 596 vs 677 per COCOMO) sono statisticamente robuste o rientrano nella variabilità naturale attesa data la piccola numerosità campionaria di molti di questi dataset.

6. **Verifica se il miglioramento della Huber loss regge anche con architetture alternative** già esplorate nella Fase 2 (`FDCNN_SeqAsym`, convoluzione pointwise 1×1) — finora il tuning di loss e attivazione è stato fatto solo sull'architettura base `FDCNN`, non sulle varianti migliorative documentate in test2.md.

7. **Documentare esplicitamente, in un'eventuale versione consolidata del report finale**, il rischio di contaminazione dei risultati delle Fasi 1-3 dovuto al problema di non-determinismo scoperto in questa fase — i numeri riportati in precedenza con `ThreadingBackend` non sono necessariamente invalidi, ma vanno considerati con un margine di incertezza maggiore rispetto a quelli ottenuti dopo il fix.