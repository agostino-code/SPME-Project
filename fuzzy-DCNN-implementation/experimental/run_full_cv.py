"""Run the complete nested CV on all 5 datasets and report timing + metrics.
Usage: python run_full_cv.py
"""
import time
import os
import numpy as np
from nested_cv import run_nested_cv

datasets_config = [
    ('../Software-estimation-datasets/albrecht.arff', 'Effort'),
    ('../Software-estimation-datasets/china.arff', 'Effort'),
    ('../Software-estimation-datasets/maxwell.arff', 'Effort'),
    ('../Software-estimation-datasets/Desharnais.csv', 'Effort'),
    ('../Software-estimation-datasets/COCOMO-81.csv', 'actual')
]

all_results = {}
all_times = {}
total_t0 = time.perf_counter()

for path, target_col in datasets_config:
    ds_name = os.path.basename(path).split('.')[0].upper()
    print("\n" + "#" * 80)
    print(f"RUNNING NESTED CV FOR: {ds_name}")
    print("#" * 80)
    
    t0 = time.perf_counter()
    results = run_nested_cv(
        dataset_path=path,
        target_column=target_col,
        n_outer_folds=5,
        n_trials=10,
        is_loo=False
    )
    elapsed = time.perf_counter() - t0
    all_results[ds_name] = results
    all_times[ds_name] = elapsed
    print(f"\n>>> {ds_name} completed in {elapsed:.1f}s")

total_elapsed = time.perf_counter() - total_t0

print("\n" + "=" * 80)
print("COMPLETE RESULTS SUMMARY")
print("=" * 80)
print(f"{'Dataset':<15} | {'MAE':<10} | {'RMSE':<10} | {'MdAE':<10} | {'MMRE':<10} | {'MdMRE':<10} | {'Pred(25)':<10} | {'Time(s)':<10}")
print("-" * 100)
for ds_name, res in all_results.items():
    t = all_times[ds_name]
    print(f"{ds_name:<15} | {res[0]:<10.4f} | {res[1]:<10.4f} | {res[2]:<10.4f} | {res[3]:<10.4f} | {res[4]:<10.4f} | {res[5]:<9.2f}% | {t:<10.1f}")
print("=" * 100)
print(f"\nTotal wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
