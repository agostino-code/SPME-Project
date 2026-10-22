# Test 1: Validazione dei Cluster Fuzzy (Fuzzy C-Means)

## Descrizione
Questo modulo sperimenta e valida diverse metriche di cluster validation geometriche per l'algoritmo Fuzzy C-Means (FCM), allo scopo di superare i limiti dell'indice di Xie-Beni (XB) su dati scalati:
- **Silhouette Score:** Misura la separazione e compattezza dei cluster hard (range $[-1, 1]$, più alto è migliore).
- **Davies-Bouldin Index (DB):** Valuta la similarità tra cluster differenti (più basso è migliore).
- **Calinski-Harabasz Index (CH):** Rapporto di dispersione tra/intra cluster (più alto è migliore).
- **Xie-Beni Index (XB):** Metrica fuzzy classica (rapporto tra variazione totale e separazione minima).
- **Validazione End-to-End LOOCV:** Verifica se il $K$ ottimale secondo le metriche geometriche minimizza effettivamente la MedAE/MAE del modello neurale F-DCNN.

## Esecuzione
Esecuzione dello script di test dedicato:
```bash
python tests/clusters/run_clusters_test.py
```

## Parametri Sperimentati
- Numero di cluster $K \in [2, 10]$
- Esponente di fuzzificazione $M \in [1.5, 2.0, 2.5]$
- Dataset valutati: *COCOMO-81*, *Albrecht*, *Desharnais*, *Maxwell*, *China*

## File di Output
- `cluster_metrics.csv`: Tabella completa con tutte le metriche calcolate per combinazione di dataset, $K$, $M$ ed errore end-to-end.
- `recommendations.json`: File strutturato JSON contenente la raccomandazione del miglior $K$ per ciascun dataset secondo ogni metrica geometrica ed empirica.
- `cluster_metrics_fab.csv` & `recommendations_fab.json`: Dati sperimentali comparativi indipendenti.
