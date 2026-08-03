# Test 3: Hyperparameter Grid Search

## Description
This test optimizes hyperparameter selection per dataset using an efficient 2-stage search strategy:
- **Stage 1 (Architecture & Regularization):** Tunes `num_filters` [8, 16, 32], `dropout_rate` [0.1, 0.2, 0.3], `weight_decay` [0.0, 0.001].
- **Stage 2 (Optimization & Training):** Tunes `dcnn_lr` [0.005, 0.001, 0.0005, 0.0001], `dcnn_batch_size` [8, 16, 32], `dcnn_epochs` [200, 500, 1000].

## Usage
Run the experiment:
```bash
python tests/gridsearch/run_gridsearch_test.py
```
Output:
- `grid_results_{dataset}.csv`: Full grid trial evaluations per dataset.
- `best_hyperparameters.json`: Consolidated optimal hyperparameter configuration per dataset.
