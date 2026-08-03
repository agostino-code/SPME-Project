# Test 4: Huber Loss Evaluation

## Description
This test evaluates the effectiveness of Huber loss (Smooth L1) and MAE loss against standard RMSE loss across all datasets:
- **RMSE Loss:** Quadratic loss, sensitive to project effort outliers.
- **MAE Loss:** Linear loss, robust to outliers.
- **Huber Loss ($\delta \in [0.01, 0.05, 0.1, 0.2, 0.5]$):** Quadratic for errors below $\delta$, linear for errors above $\delta$.

## Metrics
- MAE & MedAE
- Standardized Accuracy (SA)
- Effect Size ($\Delta$)
- Wilcoxon signed-rank test vs RMSE baseline (p-value < 0.05)

## Usage
Run the experiment:
```bash
python tests/huber_loss/run_huber_loss_test.py
```
Output:
- `huber_results.csv`: Multi-seed evaluation results for loss functions.
