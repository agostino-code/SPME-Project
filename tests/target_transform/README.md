# Test 5: Target Log-Transformation Evaluation

## Description
This test evaluates target variable log-transformation ($\log(1+y)$) to stabilize variance and handle high dynamic range/skewness in software effort datasets:
- **Linear Target Scaling:** Standard min-max scaling directly on target $y$.
- **Log Target Transformation ($\log(1+y)$):** Min-max scaling on $\log(1+y)$ during training, with $\exp(\hat{y}) - 1$ during inference.

## Usage
Run the experiment:
```bash
python tests/target_transform/run_target_transform_test.py
```
Output:
- `log_transform_comparison.csv`: Performance comparison between linear and log-scaled target models.
