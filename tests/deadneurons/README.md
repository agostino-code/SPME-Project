# Test 2: Analisi dei Neuroni Morti e Funzioni di Attivazione di Output

## Descrizione
Questo modulo analizza e confronta 5 strategie di attivazione del layer di output per eliminare il fenomeno di saturazione dei neuroni ("dying ReLU") senza introdurre distorsioni o bias sistematici nella stima dell'effort software:
- **Variante A (ReLU Standard):** `nn.ReLU()` allo strato di uscita (soggetta a collasso permanente a gradiente nullo).
- **Variante B (Softplus):** `nn.Softplus()` allo strato di uscita (garantisce gradiente continuo, ma introduce un bias positivo sistematico per valori vicini a zero).
- **Variante C (Linear + Clipping in Inferenza - Baseline Consigliato):** Output lineare puro durante il training + vincolo di non-negatività `np.maximum(preds, 0.0)` applicato esclusivamente in fase di inferenza.
- **Variante D (LeakyReLU + Clipping):** `nn.LeakyReLU(negative_slope=0.01)` in output combinato con clipping non-negativo.
- **Variante E (ELU + 1):** `nn.ELU() + 1.0` per preservare gradienti fluidi con uscite non negative.

## Metriche di Valutazione
- **Dead Neuron Rate (% Zero Preds):** Percentuale di predizioni stazionarie sul valore minimo di training (0.0).
- **MAE (Mean Absolute Error) & MedAE (Median Absolute Error)**.
- **SA (Standardized Accuracy):** Guadagno rispetto a un baseline di stima casuale permutato 1000 volte.
- **Test dei ranghi con segno di Wilcoxon ($p < 0.05$):** Verifica della significatività statistica rispetto al baseline.

## Esecuzione
Esecuzione dello script di test dedicato:
```bash
python tests/deadneurons/run_deadneurons_test.py
```

## File di Output
- `activation_comparison.csv`: Riepilogo dettagliato multi-seed su 5 seed casuali (42, 123, 456, 789, 999) per le 5 varianti su tutti i dataset.
