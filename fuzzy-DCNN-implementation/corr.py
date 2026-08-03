import sys
import pandas as pd
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python corr.py <csv_file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
corr = np.corrcoef(df['actual'], df['predicted'])[0, 1]
print(f"Correlation: {corr:.4f}")
