# Test 2: Dead Neuron Analysis

## Description
This test analyzes output activation strategies to eliminate "dying ReLU" issues without introducing positive bias:
- **Variant A (ReLU):** `nn.ReLU()` at output.
- **Variant B (Softplus):** `nn.Softplus()` at output.
- **Variant C (Linear + Clip):** Linear output during training, `np.maximum(preds, 0.0)` in inference (Baseline).
- **Variant D (LeakyReLU + Clip):** `nn.LeakyReLU(0.01)` at output + clipping.
- **Variant E (ELU + 1):** `nn.ELU() + 1.0` at output.

## Metrics
- Dead Neuron Rate (% zero predictions)
- MAE & MedAE
- Standardized Accuracy (SA)
- Wilcoxon signed-rank test vs baseline (p-value < 0.05)

## Usage
Run the experiment:
```bash
python tests/deadneurons/run_deadneurons_test.py
```
Output:
- `activation_comparison.csv`: Summary of activation performance across multi-seed runs.
