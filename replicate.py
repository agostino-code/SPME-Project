"""
Replicates methodology from:
"Assessing the effectiveness of approximate functional sizing approaches for effort estimation"
(Di Martino, Ferrucci, Gravino et al., Information and Software Technology, 2020)

This script loads the China dataset, computes approximate functional sizing metrics,
trains Simple Linear Regression (SLR) models using 10-Fold Cross-Validation,
and evaluates performance using the Mean of Absolute Residuals (MAR).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

def run_replication():
    # 1. Load the dataset
    # The China dataset contains 499 projects measured under the IFPUG FPA standard.
    df = pd.read_csv('software-cost-estimation/china.csv')
    
    # 2. Extract target variable
    # 'Effort' represents the actual development effort in person-hours (dependent variable y).
    y = df['Effort'].values
    
    # 3. Calculate Unadjusted Function Points (UFP)
    # UFP is the sum of the five standard Base Functional Components (BFCs):
    # - Input: External Inputs (EI)
    # - Output: External Outputs (EO)
    # - Enquiry: External Enquiries (EQ)
    # - File: Internal Logical Files (ILF)
    # - Interface: External Interface Files (EIF)
    # Formula: UFP = Input + Output + Enquiry + File + Interface
    df['UFP'] = df['Input'] + df['Output'] + df['Enquiry'] + df['File'] + df['Interface']
    
    # 4. Estimate component counts from functional size values
    # In the China dataset, the values in columns (Input, Output, Enquiry, File, Interface) 
    # represent the total functional size contribution of those components, not their raw counts.
    # To approximate raw counts, we divide each component's total size by the standard average 
    # complexity weight specified by IFPUG:
    # - Internal Logical Files (ILF): Average complexity weight = 10
    # - External Interface Files (EIF): Average complexity weight = 7
    # - External Inputs (EI): Average complexity weight = 4
    # - External Outputs (EO): Average complexity weight = 5
    # - External Enquiries (EQ): Average complexity weight = 4
    df['Count_ILF'] = df['File'] / 10.0
    df['Count_EIF'] = df['Interface'] / 7.0
    df['Count_EI'] = df['Input'] / 4.0
    df['Count_EO'] = df['Output'] / 5.0
    df['Count_EQ'] = df['Enquiry'] / 4.0
    
    # 5. Calculate Approximate Functional Sizing Metrics
    #
    # A. High-Level FPA (HLFPA):
    # Re-aggregates the estimated component counts by multiplying them by their respective 
    # average complexity weights:
    # Formula: HLFPA = (Count_ILF * 7) + (Count_EIF * 5) + (Count_EI * 4) + (Count_EO * 5) + (Count_EQ * 4)
    # (Note: Standard HLFPA assigns average weights of ILF=7, EIF=5, EI=4, EO=5, EQ=4)
    df['HLFPA'] = df['Count_ILF']*7 + df['Count_EIF']*5 + df['Count_EI']*4 + df['Count_EO']*5 + df['Count_EQ']*4
    
    # B. Indicative FPA (IFPA):
    # Considers only Data Functions (ILF and EIF) and ignores Transaction Functions (EI, EO, EQ).
    # Uses standard indicative weights (ILF=35, EIF=15) to compensate for missing Transaction Functions.
    # Formula: IFPA = (Count_ILF * 35) + (Count_EIF * 15)
    df['IFPA'] = df['Count_ILF']*35 + df['Count_EIF']*15
    
    # 6. Calculate Baselines
    # Baseline models serve as a comparison benchmark. They predict the mean or median effort of 
    # the entire dataset, representing a simple guessing strategy without any model training.
    # - Baseline Mean MAR: Mean of the absolute difference between each project's actual effort and the overall mean effort.
    # - Baseline Median MAR: Mean of the absolute difference between each project's actual effort and the overall median effort.
    mar_mean = np.mean(np.abs(y - np.mean(y)))
    mar_median = np.mean(np.abs(y - np.median(y)))
    
    # Size metrics to evaluate
    features = ['UFP', 'HLFPA', 'IFPA', 'AFP']
    
    results = []
    results.append(('Baseline Mean', mar_mean))
    results.append(('Baseline Median', mar_median))
    
    # 7. 10-Fold Cross-Validation Setup
    # Splits the dataset into 10 equal parts (folds).
    # In each iteration, 9 folds are used to train the Simple Linear Regression (SLR) model,
    # and the remaining fold is used to generate predictions. This is repeated 10 times.
    # shuffle=True and random_state=42 ensure reproducibility of the fold splits.
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # 8. Train and Evaluate Models
    for feat in features:
        # X is the independent variable (size metric)
        X = df[[feat]].values
        model = LinearRegression()
        
        # cross_val_predict generates out-of-fold predictions for every sample in the dataset
        y_pred = cross_val_predict(model, X, y, cv=kf)
        
        # Calculate Mean of Absolute Residuals (MAR) for the predictions.
        # Formula: MAR = (1 / N) * sum(|y_actual_i - y_predicted_i|)
        mar = mean_absolute_error(y, y_pred)
        results.append((f"SLR with {feat}", mar))
        
    # 9. Print Results
    # Output format is aligned with the required task results.
    print("--- Results (MAR) ---")
    for name, mar in results:
        print(f"{name:25s}: {mar:.2f}")

if __name__ == '__main__':
    run_replication()