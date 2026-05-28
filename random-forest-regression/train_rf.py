import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def main():
    # Carica il dataset
    df = pd.read_csv("../software-cost-estimation/china.csv")
    
    # Rimuoviamo colonne non utili o che non si conoscono a priori (ID, N_effort, Duration)
    cols_to_drop = ['ID', 'N_effort', 'Duration']
    
    # Feature (X) e Target (y)
    X = df.drop(columns=cols_to_drop + ['Effort'])
    y = df['Effort']
    
    # Split dei dati (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inizializza e addestra il modello Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predizione sul test set
    y_pred = rf_model.predict(X_test)
    
    # Calcolo Metriche
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # MMRE (Mean Magnitude of Relative Error)
    # Calcolato come media di |Reale - Predetto| / Reale
    mre = np.abs(y_test - y_pred) / y_test
    mmre = np.mean(mre)
    
    # PRED(25) - Percentuale di stime con errore relativo < 25%
    pred_25 = np.mean(mre <= 0.25) * 100
    
    print("=== Risultati Random Forest ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"MMRE: {mmre:.4f}")
    print(f"PRED(25): {pred_25:.2f}%")

    # --- VISUALIZZAZIONE DEI RISULTATI ---
    plt.figure(figsize=(14, 6))

    # Grafico 1: Valori Reali vs Predetti
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolor='k')
    # Disegniamo la linea ideale dove Predetto = Reale
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predizione Perfetta')
    plt.xlabel('Effort Reale')
    plt.ylabel('Effort Predetto')
    plt.title('Performance: Reale vs Predetto')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Grafico 2: Importanza delle Feature
    plt.subplot(1, 2, 2)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(X.shape[1]), importances[indices], align="center", color='teal')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
    plt.title("Importanza delle Feature (Random Forest)")
    plt.ylabel("Peso Relativo")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # Salviamo il grafico
    output_img = "rf_visualization.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[OK] Visualizzazione grafica salvata nel file: {output_img}")

if __name__ == "__main__":
    main()
