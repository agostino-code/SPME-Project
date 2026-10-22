# Test 5: Valutazione della Trasformazione Logaritmica del Target ($\log(1+y)$)

## Descrizione
Questo modulo analizza l'impatto della trasformazione logaritmica della variabile obiettivo (Effort) per stabilizzare la varianza, comprimere l'ampio raggio dinamico e mitigare drasticamente l'influenza asimmetrica dei progetti outlier (*skewness*):

1. **Target Lineare (Min-Max):** Scalatura lineare convenzionale dell'effort $y \in [0, 1]$.
2. **Target Logaritmico ($\log(1+y)$):**
   - In fase di training: trasformazione $y_{\text{log}} = \log(1 + y)$ seguita da scalatura Min-Max.
   - In fase di inferenza: scalatura inversa seguita dalla de-trasformazione esponenziale $\hat{y} = \exp(\hat{y}_{\text{log}}) - 1$, con vincolo $\hat{y} \ge 0$.

Le combinazioni vengono testate in combinazione sia con la loss classica (RMSE) che con la loss robusta (Huber Loss, $\delta=0.01$).

## Metodologia e Protocollo
- LOOCV per *Albrecht*, *COCOMO-81*, *Desharnais*, *Maxwell*.
- 10-Fold Cross-Validation per *China*.
- Esecuzione multi-seed su 5 seed (42, 123, 456, 789, 999).
- Test di Wilcoxon per validare la significatività statistica del miglioramento.

## Esecuzione
Esecuzione dello script di test dedicato:
```bash
python tests/target_transform/run_target_transform_test.py
```

## File di Output
- `log_transform_comparison.csv`: Tabella comparativa per tutte le combinazioni (Target Lineare vs Logaritmico $\times$ RMSE vs Huber) su ciascun dataset.
- `log_transform_comparison_fab.csv`: File di replicazione sperimentale indipendente.
