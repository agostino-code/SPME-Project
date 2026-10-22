# Test 4: Valutazione delle Funzioni di Loss Robuste (Huber vs RMSE vs MAE)

## Descrizione
Questo modulo analizza la capacità di generalizzazione e la stabilità dei gradienti al variare della funzione di perdita (*Loss Function*), con particolare attenzione alla gestione dei progetti outlier e della forte asimmetria (*skewness*) tipica dei dati di stima dell'effort software:

- **RMSE Loss (Quadratic):** Standard convenzionale di regressione. Penalizza quadraticamente gli errori ($e^2$), risultando iper-sensibile a singoli progetti fuori scala.
- **MAE Loss (Linear):** Penalità lineare ($|e|$). Robusta agli outlier, ma presenta derivata discontinua in zero.
- **Huber Loss (Smooth L1):** Funzione ibrida quadratica per piccoli scostamenti ($|e| \le \delta$) e lineare per deviazioni elevate ($|e| > \delta$). Sono stati tarati diversi valori di soglia: $\delta \in [0.01, 0.05, 0.1, 0.2, 0.5]$ su scala normalizzata $[0, 1]$.

## Metriche di Valutazione
- **MAE** & **MedAE** con deviazione standard tra i fold.
- **SA (Standardized Accuracy):** Percentuale di miglioramento rispetto a un modello nullo/casuale.
- **Effect Size ($\Delta$):** Entità dell'effetto standardizzato rispetto alla varianza casuale.
- **Test dei ranghi con segno di Wilcoxon ($p < 0.05$):** Con conteggio di vittorie (*wins*), sconfitte (*losses*) e pareggi (*ties*) per fold.

## Esecuzione
Esecuzione dello script di test dedicato:
```bash
python tests/huber_loss/run_huber_loss_test.py
```

## File di Output
- `huber_results.csv`: Risultati multi-seed (5 seed casuali) aggregati per tutte le combinazioni di loss e dataset.
- `huber_results_fab.csv`: Varianti di verifica sperimentale comparativa.
