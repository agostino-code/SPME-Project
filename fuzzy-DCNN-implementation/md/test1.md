# Report di Diagnostica Fase 1 (Baseline F-DCNN)

Durante la fase di indagine per riprodurre e allineare le performance dell'architettura F-DCNN a quelle pubblicate nello studio originale (Azzeh et al. 2024), abbiamo affrontato un divario significativo di performance, in particolare sui dataset con forti skewness (come COCOMO-81 e Albrecht).

Al fine di identificare la causa di questo gap, abbiamo isolato e testato matematicamente ogni singola componente della rete e dell'algoritmo di training. Di seguito l'elenco cronologico delle ipotesi testate.

## Cronologia dei Test Diagnostici

1. **Dead ReLU e Bias Init**
   - **Problema**: La rete non iniziava il training (loss piatta, predizioni ancorate a 0).
   - **Diagnosi e Fix**: È stata implementata l'inizializzazione Kaiming per i pesi e il bias del layer di output è stato inizializzato al target medio del training set. Questo ha "sbloccato" la rete, permettendo la discesa della loss sui dataset sani.

2. **Dimensionamento Spaziale del Pooling**
   - **Problema**: Il layer di MaxPool eseguiva padding implicito che causava una dilatazione spaziale (es. 7x2 -> 8x3) anziché la riduzione prevista dal paper.
   - **Fix**: Applicato custom pooling senza padding. Il down-sampling spaziale è ora corretto e allineato alle descrizioni del paper.

3. **Sensibilità Outlier nella Validazione LOOCV**
   - **Test**: Misurazione dell'impatto dei singoli fold sulle metriche aggregate (MAE/MdAE).
   - **Risultato**: Oltre il 50% del MAE totale su COCOMO-81 deriva esclusivamente dall'incapacità della rete di predire 3 fold estremi (su 63), dimostrando l'estrema sensibilità della metrica LOOCV sui campioni molto piccoli.

4. **Isolamento Seed Worker e Varianza Stocastica**
   - **Problema**: Fasi di parallelizzazione causavano variazioni caotiche sulle traiettorie del gradiente per fold simili.
   - **Fix**: Isolamento rigoroso del seed per singolo worker/fold per garantire il totale determinismo del processo di discesa.

5. **Esplosione del Gradiente su Loss MSE**
   - **Test**: Tracciamento della norma del gradiente nei primissimi step del training.
   - **Risultato**: L'utilizzo dell'Errore Quadratico Medio (MSE) su target non scalati con estrema varianza produce un gradiente >10.000, che rischia di distruggere i pesi istantaneamente. Il `gradient clipping` tampona l'esplosione, ma spinge inesorabilmente la rete verso una saturazione lenta (Dead ReLU).

6. **Target Scaling Lineare (Min-Max)**
   - **Ipotesi**: Ridurre il target al range `[0, 1]` potrebbe sanare l'instabilità del gradiente.
   - **Risultato**: Fallimento. Nello spazio MSE, la distanza relativa tra i target compressi e l'outlier (skewness) rimane immutata. I gradienti generano la stessa forza attrattiva sbilanciata, portando comunque a un collasso generalizzato su COCOMO-81 (il 70% dei fold muore a 0.0).

7. **Target Scaling Logaritmico (Log1p)**
   - **Ipotesi**: Comprimere matematicamente lo skewness riducendo l'outlier a un valore gestibile.
   - **Risultato**: Il collasso Dead ReLU viene curato perfettamente (il 100% dei fold di COCOMO non collassa più). Tuttavia, la rete converge a un MAE di **655** (molto distante dal 160 del paper). La rete sottostima pesantemente gli outlier.

8. **Rappresentazione FCM (Fuzzy C-Means)**
   - **Ipotesi**: Il clustering FCM "perde" il segnale se l'outlier è troppo distante dai centroidi calcolati sul training set.
   - **Risultato**: Nessuna conferma chiara. Il metodo di confronto tra un singolo campione di test e la media dei campioni di training non è probante (la media tende a distribuirsi uniformemente per natura). Servirebbero analisi più profonde sulla distribuzione individuale delle membership, ma il rapporto costi/benefici diagnostici suggerisce di fermarsi.

---

## Limiti della Replica

Nel definire l'esito della diagnosi, bisogna riconoscere l'impossibilità di verificare alcuni parametri critici totalmente fuori dal nostro controllo:
* Non abbiamo accesso al codice sorgente originale.
* Non sappiamo se le versioni dei dataset usate dagli autori includessero una "pulizia" (rimozione) pregressa degli outlier estremi, pratica frequente ma non esplicitata nel testo.
* Ignoriamo l'eventuale adozione di trasformazioni di scaling (sulle feature o sul target) occultate nel report metodologico del paper.

## Conclusione Diagnostica Definitiva

Non siamo riusciti a riprodurre il MAE dichiarato sul dataset COCOMO-81 (160) né su Albrecht (1.1). Le instabilità numeriche intrinseche all'architettura sono state tutte isolate e mitigate al meglio, senza deviare dalle scelte matematiche descritte nello studio (in primis l'uso della loss MSE su target puro).

Il gap residuo è plausibilmente dovuto a dettagli metodologici e implementativi (gestione degli outlier, trasformazioni non lineari del target) non dichiarati nel paper originale, ma questa rimane una solida ipotesi tecnica, non una prova verificata sul loro ambiente chiuso.
A livello strutturale, l'architettura esatta descritta nel paper incontra dei limiti oggettivi se incaricata di estrapolare target iper-sbilanciati basandosi su campioni microscopici.

---

## Tabella Finale (Stato Baseline F-DCNN)

| Dataset | MAE Nostro | MAE Paper | Ratio (Nostro/Paper) | Stato |
| :--- | :--- | :--- | :--- | :--- |
| **China** | 1519.79 | 899 | ~1.69x | Vicino al paper (gap accettabile su ~500 righe) |
| **Desharnais**| 2277.98 | 1044 | ~2.18x | Gap moderato, parzialmente coerente con campionamento LOOCV |
| **Maxwell** | 6438.75 | 1749 | ~3.68x | Gap sensibile, dominato da dimensione ridotta |
| **COCOMO-81** | 755.18 | 160 | ~4.72x | Gap ampio, causa parzialmente identificata (skew/MSE) ma irrisolta |
| **Albrecht** | 8.65 | 1.1 | ~7.86x | Gap più ampio in assoluto, causa non identificata, problema aperto |

*(Tutti i valori nella colonna "MAE Paper" sono stati controllati e verificati contro la Table 3 del paper originale di Azzeh et al. 2024)*
