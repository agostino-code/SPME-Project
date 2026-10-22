#!/usr/bin/env python3
"""
Test Suite: Baseline Comparison Models for SDEE
Evaluates:
1. Di Martino et al. (Simple Linear Regression / SLR on primary size metric)
2. Random Forest Regressor (RF)
3. Support Vector Regression (SVR / SVM)
4. Comparison against our optimized Fuzzy-DCNN

Protocols:
- LOOCV (Leave-One-Out Cross-Validation) for small datasets: Albrecht, COCOMO-81, Desharnais, Maxwell
- 10-Fold Cross-Validation for China
"""

import os
import sys
import numpy as np
import pandas as pd

# Fix seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASETS_DIR = os.path.join(BASE_DIR, 'Software-estimation-datasets')

# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------
def load_arff(filepath):
    cols, data, in_data = [], [], False
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            l = line.strip()
            if not l or l.startswith('%'):
                continue
            if l.lower().startswith('@attribute'):
                cols.append(l.split()[1])
            elif l.lower().startswith('@data'):
                in_data = True
            elif in_data:
                row = []
                for x in l.split(','):
                    x = x.strip().strip("'\"")
                    try:
                        row.append(float(x) if x != '?' else np.nan)
                    except ValueError:
                        row.append(x)
                data.append(row)
    return pd.DataFrame(data, columns=cols)

def get_dataset(name):
    """Loads and preprocesses datasets matching dataset.py logic exactly."""
    if name == 'china':
        path = os.path.join(DATASETS_DIR, 'china.arff')
        df = load_arff(path)
        drop_cols = ['ID', 'Duration', 'N_effort', 'Resource', 'Dev.Type']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna().reset_index(drop=True)
        size_col = 'AFP'
        target_col = 'Effort'
    elif name == 'cocomo':
        path = os.path.join(DATASETS_DIR, 'COCOMO-81.csv')
        df = pd.read_csv(path)
        drop_cols = ['num', 'Type', 'DEV']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        for c in df.select_dtypes(['object']).columns:
            df[c] = df[c].astype('category').cat.codes
        df = df.dropna().reset_index(drop=True)
        size_col = 'loc'
        target_col = 'actual'
    elif name == 'albrecht':
        path = os.path.join(DATASETS_DIR, 'albrecht.arff')
        df = load_arff(path).dropna().reset_index(drop=True)
        size_col = 'AdjFP'
        target_col = 'Effort'
    elif name == 'desharnais':
        path = os.path.join(DATASETS_DIR, 'Desharnais.csv')
        df = pd.read_csv(path).replace({-1: np.nan, '-1': np.nan, -1.0: np.nan})
        drop_cols = ['id', 'Project', 'YearEnd', 'Language', 'AdjFactor']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna().reset_index(drop=True)
        size_col = 'PointsAjust'
        target_col = 'Effort'
    elif name == 'maxwell':
        path = os.path.join(DATASETS_DIR, 'maxwell.arff')
        df = load_arff(path)
        drop_cols = ['App', 'Har', 'Dba', 'Ifc', 'Source', 'Telonuse']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna().reset_index(drop=True)
        size_col = 'Size'
        target_col = 'Effort'
    else:
        raise ValueError(f"Unknown dataset: {name}")

    y = df[target_col].values.astype(float)
    size = df[size_col].values.astype(float)
    X = df.drop(columns=[target_col]).select_dtypes(include=['number']).values.astype(float)

    return df, X, y, size

# ---------------------------------------------------------------------------
# Fallback / Pure NumPy Models if scikit-learn is not available
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.ensemble import RandomForestRegressor as SklearnRF
    from sklearn.svm import SVR as SklearnSVR
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class OLSLinearRegression:
    def fit(self, X, y):
        X_mat = np.column_stack([np.ones(len(X)), X])
        self.beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        return self

    def predict(self, X):
        X_mat = np.column_stack([np.ones(len(X)), X])
        preds = X_mat @ self.beta
        return np.maximum(0.0, preds)

class NumpyDecisionTree:
    def __init__(self, max_depth=6, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.all(y == y[0]):
            return {'leaf': True, 'val': float(np.mean(y))}

        feat_indices = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            feat_indices = np.random.choice(n_features, self.max_features, replace=False)

        best_feat, best_thresh, best_score = None, None, float('inf')

        for f in feat_indices:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 15:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 15))

            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                var_l = np.var(y[left_mask]) * np.sum(left_mask)
                var_r = np.var(y[right_mask]) * np.sum(right_mask)
                score = var_l + var_r
                if score < best_score:
                    best_score = score
                    best_feat = f
                    best_thresh = t

        if best_feat is None:
            return {'leaf': True, 'val': float(np.mean(y))}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            'leaf': False,
            'feat': best_feat,
            'thresh': best_thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        }

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in X])

    def _predict_one(self, row, node):
        if node['leaf']:
            return node['val']
        if row[node['feat']] <= node['thresh']:
            return self._predict_one(row, node['left'])
        return self._predict_one(row, node['right'])

class NumpyRandomForest:
    def __init__(self, n_estimators=50, max_depth=6, max_features=None, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n_samples, n_features = X.shape
        max_f = self.max_features if self.max_features else max(1, int(np.sqrt(n_features)))

        self.trees = []
        for i in range(self.n_estimators):
            sample_indices = rng.choice(n_samples, n_samples, replace=True)
            X_boot = X[sample_indices]
            y_boot = y[sample_indices]
            tree = NumpyDecisionTree(max_depth=self.max_depth, max_features=max_f)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.maximum(0.0, np.mean(all_preds, axis=0))

class NumpySVR:
    """Kernel Ridge / RBF Support Vector Regression in NumPy."""
    def __init__(self, gamma=0.1, alpha=1.0):
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X, y):
        # Normalize features
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.x_std[self.x_std == 0] = 1.0
        X_norm = (X - self.x_mean) / self.x_std

        self.X_train = X_norm
        # RBF Kernel matrix
        dists = np.sum(X_norm**2, axis=1, keepdims=True) + np.sum(X_norm**2, axis=1, keepdims=True).T - 2 * (X_norm @ X_norm.T)
        dists = np.maximum(0.0, dists)
        K = np.exp(-self.gamma * dists)
        self.dual_coef = np.linalg.solve(K + self.alpha * np.eye(len(X_norm)), y)
        return self

    def predict(self, X):
        X_norm = (X - self.x_mean) / self.x_std
        dists = np.sum(X_norm**2, axis=1, keepdims=True) + np.sum(self.X_train**2, axis=1, keepdims=True).T - 2 * (X_norm @ self.X_train.T)
        dists = np.maximum(0.0, dists)
        K_test = np.exp(-self.gamma * dists)
        preds = K_test @ self.dual_coef
        return np.maximum(0.0, preds)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_eval_metrics(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    abs_errors = np.abs(actual - predicted)
    mae = float(np.mean(abs_errors))
    medae = float(np.median(abs_errors))

    # Permutation test for Standardized Accuracy (SA)
    random_maes = []
    rng = np.random.RandomState(42)
    for _ in range(1000):
        perm_actual = rng.permutation(actual)
        random_maes.append(np.mean(np.abs(actual - perm_actual)))
    mae_random = np.mean(random_maes)
    sa = float(1.0 - (mae / mae_random)) if mae_random != 0 else 0.0

    return {
        'MAE': round(mae, 2),
        'MedAE': round(medae, 2),
        'SA': round(sa * 100.0, 1)
    }

# ---------------------------------------------------------------------------
# Evaluation Loops (LOOCV & 10-Fold CV)
# ---------------------------------------------------------------------------
def evaluate_models_on_dataset(ds_name):
    df, X, y, size = get_dataset(ds_name)
    n = len(y)
    print(f"\nEvaluating on '{ds_name.upper()}' (N={n})...")

    is_china = (ds_name == 'china')

    if is_china:
        # 10-Fold CV with fixed seed
        rng = np.random.RandomState(42)
        indices = np.arange(n)
        rng.shuffle(indices)
        folds = np.array_split(indices, 10)
    else:
        # LOOCV
        folds = [[i] for i in range(n)]

    preds_slr = np.zeros(n)
    preds_rf = np.zeros(n)
    preds_svr = np.zeros(n)

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)

        # 1. Di Martino SLR: E = beta_0 + beta_1 * Size
        size_train = size[train_idx].reshape(-1, 1)
        size_test = size[test_idx].reshape(-1, 1)
        y_train = y[train_idx]

        if HAS_SKLEARN:
            slr = SklearnLR().fit(size_train, y_train)
            preds_slr[test_idx] = np.maximum(0.0, slr.predict(size_test))
        else:
            slr = OLSLinearRegression().fit(size_train, y_train)
            preds_slr[test_idx] = slr.predict(size_test)

        # 2. Random Forest Regressor
        X_train, X_test = X[train_idx], X[test_idx]
        if HAS_SKLEARN:
            rf = SklearnRF(n_estimators=50, max_depth=6, random_state=42).fit(X_train, y_train)
            preds_rf[test_idx] = np.maximum(0.0, rf.predict(X_test))
        else:
            rf = NumpyRandomForest(n_estimators=40, max_depth=5, seed=42 + fold_idx).fit(X_train, y_train)
            preds_rf[test_idx] = rf.predict(X_test)

        # 3. SVR
        if HAS_SKLEARN:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_train)
            X_tr_s = scaler.transform(X_train)
            X_te_s = scaler.transform(X_test)
            svr = SklearnSVR(C=10.0, gamma='scale').fit(X_tr_s, y_train)
            preds_svr[test_idx] = np.maximum(0.0, svr.predict(X_te_s))
        else:
            svr = NumpySVR(gamma=0.05, alpha=1.0).fit(X_train, y_train)
            preds_svr[test_idx] = svr.predict(X_test)

    m_slr = compute_eval_metrics(y, preds_slr)
    m_rf = compute_eval_metrics(y, preds_rf)
    m_svr = compute_eval_metrics(y, preds_svr)

    return {
        'SLR': m_slr,
        'RF': m_rf,
        'SVR': m_svr
    }

# ---------------------------------------------------------------------------
# SOTA Fuzzy-DCNN Results (from thesis / risultati_test_completi.md)
# ---------------------------------------------------------------------------
OUR_FDCNN_RESULTS = {
    'china': {'MAE': 817.22, 'MedAE': 117.09, 'SA': 83.0},
    'cocomo': {'MAE': 588.67, 'MedAE': 71.24, 'SA': 51.0},
    'albrecht': {'MAE': 9.72, 'MedAE': 3.11, 'SA': 76.0},
    'desharnais': {'MAE': 2255.37, 'MedAE': 995.24, 'SA': 50.0},
    'maxwell': {'MAE': 3823.85, 'MedAE': 1986.93, 'SA': 56.0}
}

DATASET_NAMES = ['china', 'cocomo', 'albrecht', 'desharnais', 'maxwell']

def main():
    print("=" * 80)
    print("RUNNING BASELINES EVALUATION (Di Martino SLR, Random Forest, SVR)")
    print(f"Backend: {'scikit-learn' if HAS_SKLEARN else 'Pure NumPy (Reproducible Engine)'}")
    print("=" * 80)

    all_records = []

    for ds in DATASET_NAMES:
        res = evaluate_models_on_dataset(ds)
        fdcnn = OUR_FDCNN_RESULTS[ds]

        for model_name, metrics in [('Di Martino (SLR)', res['SLR']),
                                    ('Random Forest', res['RF']),
                                    ('SVM (SVR)', res['SVR']),
                                    ('Nostro Fuzzy-DCNN (Ottimizzato)', fdcnn)]:
            all_records.append({
                'Dataset': ds.upper(),
                'Modello': model_name,
                'MAE': metrics['MAE'],
                'MedAE': metrics['MedAE'],
                'SA (%)': metrics['SA']
            })

    df_out = pd.DataFrame(all_records)
    out_csv = os.path.join(os.path.dirname(__file__), 'baseline_comparison_results.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")

    # Display Pivot / Summary Table
    print("\n" + "=" * 80)
    print("TABELLA COMPARATIVA COMPLETA: BASELINES VS NOSTRO FUZZY-DCNN")
    print("=" * 80)
    pivot_medae = df_out.pivot(index='Dataset', columns='Modello', values='MedAE')
    print("\n--- MEDIAN ABSOLUTE ERROR (MedAE) - Più basso è migliore ---")
    print(pivot_medae.to_string())

    pivot_mae = df_out.pivot(index='Dataset', columns='Modello', values='MAE')
    print("\n--- MEAN ABSOLUTE ERROR (MAE) ---")
    print(pivot_mae.to_string())

    pivot_sa = df_out.pivot(index='Dataset', columns='Modello', values='SA (%)')
    print("\n--- STANDARDIZED ACCURACY (SA %) - Più alto è migliore ---")
    print(pivot_sa.to_string())

run_baselines_comparison = main

if __name__ == '__main__':
    main()
