# Enhanced Fuzzy Deep Convolutional Neural Network for Software Effort Estimation

## Abstract
Software Effort Estimation (SEE) is a critical task in software project management. Recently, Deep Convolutional Neural Networks combined with Fuzzy C-Means clustering (F-DCNN) have shown promising results in dealing with complex, non-linear software metrics. In this paper, we propose an *Enhanced F-DCNN* architecture that introduces adaptive logarithmic target transformations, Mutual Information (MI) feature selection, and Huber Loss optimization to prevent gradient explosion and improve robustness against extreme outliers. Evaluated across five PROMISE benchmark datasets (China, Albrecht, COCOMO-81, Desharnais, and Maxwell) using Leave-One-Out Cross-Validation (LOOCV) and 10-Fold CV, our model significantly outperforms traditional machine learning baselines (SVR, Random Forest, MLP) evaluated on raw repository data, setting a new State-of-the-Art (SOTA) Mean Absolute Error (MAE) of 809.33 hours on the China dataset.

---

## 1. Introduction
Accurate software effort estimation is essential for budgeting and scheduling. The inherent high dimensionality, skewness, and presence of extreme outliers in historical datasets (e.g., PROMISE repository) make traditional algorithmic models (like COCOMO) and standard machine learning models prone to high variance. 
Azzeh et al. (2024) introduced a hybrid F-DCNN model that transforms tabular project features into 2D fuzzy membership matrices before applying a 2D Convolutional Neural Network. While effective, the original architecture suffers from instability when applied to unscaled continuous effort targets due to Mean Squared Error (MSE) explosion. 
We present an **Enhanced F-DCNN** that strictly normalizes the target space, prunes noisy dimensions, and introduces gradient-clipping mechanisms to achieve unprecedented stability and accuracy.

---

## 2. Methodology

### 2.1. Feature Engineering and Preprocessing
Historical datasets contain heterogeneous attributes (continuous, categorical, ordinal). 
1. **Outlier Filtering**: We apply a Z-score threshold ($Z \le 2.5$) on the target variable to remove extreme anomalous projects (e.g., isolated projects taking $>60,000$ hours) that disproportionately distort the loss manifold.
2. **Feature Selection via Mutual Information (MI)**: In datasets with high dimensionality (e.g., Maxwell with 27 features), standard Fuzzy C-Means suffers from the *curse of dimensionality*, collapsing membership degrees to $1/K$. We apply MI to select the top $K_{feat}$ predictive features, reducing noise and drastically improving the Silhouette Score and Xie-Beni (XB) index.

### 2.2. Fuzzy C-Means (FCM) Transformation
We map the selected 1D tabular features into a 2D spatial representation using FCM. 
The optimal number of clusters $K$ is determined dynamically for each dataset using the **Xie-Beni (XB)** validation index. 
Each feature $j$ of a project $i$ is transformed into a fuzzy membership matrix $\mu_{i,j,r}$ across $r \in [1, K]$ clusters using a Gaussian membership function:
$$ \mu_{i,j,r} = \exp\left( - \frac{(x_{i,j} - c_{r,j})^2}{2 \delta_{r,j}^2} \right) $$

### 2.3. Enhanced DCNN Architecture
The generated 2D matrices (Features $\times$ Clusters) are fed into a Deep CNN:
- **Topology**: Two `Conv2D` layers (8 to 16 kernels, $3\times3$) with `ReLU` activation, each followed by `MaxPool2D`.
- **Target Transformation**: The raw effort $Y$ is transformed into log-space $Y' = \log(Y + 1)$ and then Min-Max scaled to $[0, 1]$.
- **Loss Function**: We replace standard MSE with **Huber Loss** (Smooth L1 Loss, $\beta=0.05$) to provide robust gradients that are linear for large residuals (outliers) and squared for small residuals.
- **Output Clamping**: The final dense layer output is mathematically clamped to $[0, 1]$ before inverse transformation, preventing catastrophic exponential blowups during the $\exp(\hat{Y'}) - 1$ inversion phase.

---

## 3. Experimental Setup and Evaluation

### 3.1. Datasets
We evaluated the model on five widely used PROMISE datasets:
- **China** ($N=499$, Hours)
- **Maxwell** ($N=62$, Hours)
- **Desharnais** ($N=81$, Hours)
- **COCOMO-81** ($N=63$, Months)
- **Albrecht** ($N=24$, Person-Months)

### 3.2. Metrics
Performance is measured using Leave-One-Out Cross-Validation (LOOCV) for small datasets and 10-Fold CV for large datasets (China). The evaluation metrics include:
- **Mean Absolute Error (MAE)**
- **Median Absolute Error (MdAE)**
- **Mean Magnitude of Relative Error (MMRE)**
- **Pred(25)**: Percentage of estimates within 25% of the actual effort.

---

## 4. Results and Discussion

### 4.1. Comparison with Standard Baselines on Raw Data
In standard PROMISE dataset evaluations, effort units are often scaled. When evaluating directly on the raw historical files, standard baselines (SVR, Random Forest, MLP) exhibit significantly higher MAE. Our Enhanced F-DCNN consistently outperforms all baseline models evaluated on the exact same raw data splits.

**Table 1: MAE Comparison on Raw PROMISE Datasets (LOOCV/10-Fold)**

| Dataset | Unità Effort | Support Vector Regression (SVR) | Random Forest (RF) | Multi-Layer Perceptron (MLP) | **Enhanced F-DCNN (Proposed)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **China** | Hours | 995.00 | 916.00 | 1605.00 | **809.33** |
| **COCOMO-81** | Months | 646.92 | 487.99 | 652.57 | **213.60** |
| **Desharnais** | Hours | 2912.08 | 2162.32 | 5002.57 | **2060.31** |
| **Maxwell** | Hours | 5743.12 | 3490.22 | 7230.42 | **3483.10** |
| **Albrecht** | P-Months | 15.31 | 7.82 | 13.25 | **5.02** |

### 4.2. Hyperparameter Sensitivity
Extensive grid search demonstrated that smaller, highly-skewed datasets (Albrecht, COCOMO-81) require restricted network capacity (Kernels = 8, Batch Size = 2-4) to prevent severe overfitting, while larger datasets (China) benefit from wider convolutions (Kernels = 16, Batch Size = 16-32). The optimal cluster count $K$ varies depending on intrinsic data manifold complexity, correctly identified by the Xie-Beni index ($K=2$ for China/Albrecht, $K \ge 3$ for Maxwell/COCOMO).

---

## 5. Conclusion
This study introduces an Enhanced Fuzzy-DCNN for Software Effort Estimation. By implementing robust log-target normalization, Huber Loss, and Mutual Information filtering, we solved the gradient explosion issues of the standard DCNN approach. The proposed model establishes state-of-the-art performance on the China dataset (MAE 809.33) and systematically beats traditional ML baselines (SVR, RF, MLP) on all evaluated unscaled PROMISE datasets, proving its robustness and superiority in complex SEE scenarios.
