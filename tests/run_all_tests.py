import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

from clusters.run_clusters_test import run_cluster_validation
from deadneurons.run_deadneurons_test import run_dead_neuron_analysis
from gridsearch.run_gridsearch_test import run_2stage_gridsearch
from huber_loss.run_huber_loss_test import run_huber_loss_evaluation
from target_transform.run_target_transform_test import run_target_transformation_test

def main():
    print("=" * 90)
    print("      FUZZY-DCNN COMPLETE SOTA TEST SUITE RUNNER")
    print("=" * 90)
    start_total = time.time()

    print("\n>>> STEP 1/5: RUNNING TEST 1 (CLUSTER VALIDATION)...")
    run_cluster_validation()

    print("\n>>> STEP 2/5: RUNNING TEST 2 (DEAD NEURON ANALYSIS)...")
    run_dead_neuron_analysis()

    print("\n>>> STEP 3/5: RUNNING TEST 3 (2-STAGE GRID SEARCH)...")
    run_2stage_gridsearch()

    print("\n>>> STEP 4/5: RUNNING TEST 4 (HUBER LOSS EVALUATION)...")
    run_huber_loss_evaluation()

    print("\n>>> STEP 5/5: RUNNING TEST 5 (TARGET LOG-TRANSFORMATION)...")
    run_target_transformation_test()

    total_time = time.time() - start_total
    print("\n" + "=" * 90)
    print(f"ALL 5 TESTS COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 90)

if __name__ == '__main__':
    main()
