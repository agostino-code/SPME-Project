import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

class Dataset:

    def __init__(self, arff_path, target_column_name='Effort', test_indices=None, train_indices=None, top_k_features=None, apply_skew_transform=True, max_corr_threshold=0.98, use_pca=False, pca_variance=0.95, augment_rare_targets=False, enable_feature_engineering=True):
        if str(arff_path).endswith('.csv'):
            data = pd.read_csv(arff_path)
        else:
            from scipy.io import arff
            raw_data, meta = arff.loadarff(arff_path)
            data = pd.DataFrame(raw_data)
            for col in data.select_dtypes([object]).columns:
                data[col] = data[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)



        # Dataset-specific cleaning
        ds_name = str(arff_path).lower()
        if 'china' in ds_name:
            drop_cols = ['ID', 'Duration', 'N_effort', 'Resource', 'PDR_AFP', 'PDR_UFP', 'NPDR_AFP', 'NPDU_UFP']
            data = data.drop(columns=[c for c in drop_cols if c in data.columns])
        elif 'desharnais' in ds_name:
            data = data.replace({-1: np.nan, "-1": np.nan, -1.0: np.nan})
            if 'Language' in data.columns:
                data['Language'] = data['Language'].astype(str)
            drop_cols = ['id', 'Project']
            data = data.drop(columns=[c for c in drop_cols if c in data.columns])
        elif 'cocomo' in ds_name:
            if 'num' in data.columns:
                data = data.drop(columns=['num'])

        # Separate features and target
        target = data[target_column_name].astype(float)
        features = data.drop(columns=[target_column_name])

        # Initial Train/Test Split BEFORE any statistical fitting (Prevents Data Leakage)
        if train_indices is not None and test_indices is not None:
            train_idx = features.index[train_indices]
            test_idx = features.index[test_indices]
        elif test_indices is None:
            train_idx = features.sample(frac=0.7, random_state=42).index
            test_idx = features.index.difference(train_idx)
        else:
            test_idx = features.index[test_indices]
            train_idx = features.index.difference(test_idx)

        training_features = features.loc[train_idx].copy()
        testing_features = features.loc[test_idx].copy()

        training_target = target.loc[train_idx].astype(float).copy()
        testing_target = target.loc[test_idx].astype(float).copy()

        # Outlier filtering EXCLUSIVELY on training target (Leakage Fix)
        z_scores = np.abs((training_target - training_target.mean()) / (training_target.std() + 1e-8))
        clean_mask = z_scores <= 2.5
        training_features = training_features[clean_mask].reset_index(drop=True)
        training_target = training_target[clean_mask].reset_index(drop=True)

        testing_features = testing_features.reset_index(drop=True)
        testing_target = testing_target.reset_index(drop=True)

        # SMOGN-style Synthetic Data Augmentation on Rare Extreme Targets (Train set strictly)
        if augment_rare_targets and len(training_target) > 15:
            high_thresh = float(np.quantile(training_target, 0.85))
            high_idx = np.where(training_target.values >= high_thresh)[0]
            if len(high_idx) >= 2:
                num_cols_list = list(training_features.select_dtypes(include=['number']).columns)
                synth_feats = []
                synth_targs = []
                for _ in range(min(12, len(high_idx) * 2)):
                    i1, i2 = np.random.choice(high_idx, size=2, replace=False)
                    lam = np.random.uniform(0.25, 0.75)
                    f_synth = training_features.iloc[i1].copy()
                    for c in num_cols_list:
                        v1 = float(training_features.iloc[i1][c])
                        v2 = float(training_features.iloc[i2][c])
                        f_synth[c] = lam * v1 + (1.0 - lam) * v2 + np.random.normal(0, 0.01)

                    t_synth = float(lam * training_target.iloc[i1] + (1.0 - lam) * training_target.iloc[i2])
                    synth_feats.append(f_synth)
                    synth_targs.append(t_synth)

                df_synth = pd.DataFrame(synth_feats)
                s_synth = pd.Series(synth_targs)
                training_features = pd.concat([training_features, df_synth], ignore_index=True)
                training_target = pd.concat([training_target, s_synth], ignore_index=True)

        # Separate Numerical and Categorical columns
        num_cols = list(training_features.select_dtypes(include=['number']).columns)
        cat_cols = list(training_features.select_dtypes(exclude=['number']).columns)

        # Non-linear Feature Engineering (Domain Ratios & Interaction Products)
        if enable_feature_engineering and len(num_cols) >= 2:
            num_cols_subset = num_cols[:4]
            for i in range(len(num_cols_subset)):
                for j in range(i + 1, len(num_cols_subset)):
                    c1, c2 = num_cols_subset[i], num_cols_subset[j]
                    denom_tr = np.abs(training_features[c1].values) + np.abs(training_features[c2].values) + 1.0
                    denom_te = np.abs(testing_features[c1].values) + np.abs(testing_features[c2].values) + 1.0
                    training_features[f"{c1}_div_{c2}"] = training_features[c1].values / denom_tr
                    testing_features[f"{c1}_div_{c2}"] = testing_features[c1].values / denom_te
                    training_features[f"{c1}_mult_{c2}"] = training_features[c1].values * training_features[c2].values
                    testing_features[f"{c1}_mult_{c2}"] = testing_features[c1].values * testing_features[c2].values

            num_cols = list(training_features.select_dtypes(include=['number']).columns)
            training_features[num_cols] = training_features[num_cols].replace([np.inf, -np.inf], np.nan)
            testing_features[num_cols] = testing_features[num_cols].replace([np.inf, -np.inf], np.nan)


        # Imputation on Numerical Features (KNNImputer fitted on Train only)
        if num_cols:
            imputer = KNNImputer(n_neighbors=min(5, max(1, len(training_features) - 1)))
            training_num = pd.DataFrame(imputer.fit_transform(training_features[num_cols]), columns=num_cols)
            testing_num = pd.DataFrame(imputer.transform(testing_features[num_cols]), columns=num_cols)
        else:
            training_num = pd.DataFrame()
            testing_num = pd.DataFrame()


        # Encoding Categorical Features (OneHotEncoder fitted on Train only)
        if cat_cols:
            training_features[cat_cols] = training_features[cat_cols].astype(str)
            testing_features[cat_cols] = testing_features[cat_cols].astype(str)
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            train_cat_enc = ohe.fit_transform(training_features[cat_cols])
            test_cat_enc = ohe.transform(testing_features[cat_cols])

            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            training_cat = pd.DataFrame(train_cat_enc, columns=cat_feature_names)
            testing_cat = pd.DataFrame(test_cat_enc, columns=cat_feature_names)

            training_features = pd.concat([training_num, training_cat], axis=1)
            testing_features = pd.concat([testing_num, testing_cat], axis=1)
        else:
            training_features = training_num
            testing_features = testing_num

        # Skewness Handling on Independent Features (log1p transform for right-skewed features > 1.0)
        if apply_skew_transform:
            for col in num_cols:
                if col in training_features.columns:
                    if training_features[col].min() >= 0 and skew(training_features[col].dropna()) > 1.0:
                        training_features[col] = np.log1p(np.maximum(0.0, training_features[col]))
                        testing_features[col] = np.log1p(np.maximum(0.0, testing_features[col]))

        # Multicollinearity / Redundancy Filter (Correlation Threshold fitted on Train)
        if max_corr_threshold is not None and training_features.shape[1] > 1:
            corr_matrix = training_features.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > max_corr_threshold)]
            if to_drop and len(to_drop) < training_features.shape[1]:
                training_features = training_features.drop(columns=to_drop)
                testing_features = testing_features.drop(columns=to_drop)

        # Ensure no NaNs remain before PCA or Feature Selection
        training_features = training_features.fillna(training_features.mean()).fillna(0.0)
        testing_features = testing_features.fillna(training_features.mean()).fillna(0.0)

        # Dimensionality Reduction via PCA (Pre-Clustering Orthogonal Topology)
        if use_pca and training_features.shape[1] > 3:
            n_comp = min(pca_variance if isinstance(pca_variance, float) else int(pca_variance), training_features.shape[1] - 1)
            pca = PCA(n_components=n_comp, random_state=42)
            tr_pca = pca.fit_transform(training_features)
            te_pca = pca.transform(testing_features)
            pca_cols = [f"PC_{i+1}" for i in range(tr_pca.shape[1])]
            training_features = pd.DataFrame(tr_pca, columns=pca_cols)
            testing_features = pd.DataFrame(te_pca, columns=pca_cols)


        # Dynamic Feature Selection via Mutual Information
        if top_k_features is not None and top_k_features < training_features.shape[1]:
            mi = mutual_info_regression(training_features.values, np.log1p(training_target.values), random_state=42)
            top_cols = training_features.columns[np.argsort(mi)[::-1][:top_k_features]]
            training_features = training_features[top_cols]
            testing_features = testing_features[top_cols]


        # Min-Max Scaling (Fit on Train only)
        self.feature_min = training_features.min(axis=0)
        self.feature_max = training_features.max(axis=0)
        feature_range = self.feature_max - self.feature_min
        feature_range[feature_range == 0] = 1.0

        self.training_data = ((training_features - self.feature_min) / feature_range).values.T
        self.testing_data = ((testing_features - self.feature_min) / feature_range).values.T

        # Adaptive Log-Transform on Target
        self.use_log_target = True if float(training_target.max()) > 200.0 else False
        if self.use_log_target:
            training_target_proc = np.log1p(training_target)
            testing_target_proc = np.log1p(testing_target)
        else:
            training_target_proc = training_target.copy()
            testing_target_proc = testing_target.copy()

        self.target_min = float(training_target_proc.min())
        self.target_max = float(training_target_proc.max())
        self.target_range = self.target_max - self.target_min if self.target_max != self.target_min else 1.0

        self.training_target_scaled = ((training_target_proc - self.target_min) / self.target_range)
        self.testing_target_scaled = ((testing_target_proc - self.target_min) / self.target_range)

        self.data = features
        self.training_features_df = training_features
        self.testing_features_df = testing_features

    def inverse_transform_target(self, y_scaled):
        y_scaled = np.asarray(y_scaled, dtype=float)
        y_unscaled = (y_scaled * self.target_range) + self.target_min
        if self.use_log_target:
            return np.expm1(y_unscaled)
        return y_unscaled

    def print_info(self):
        print("Dataset Info:")
        print(self.data.info())
        print("\nTraining Target Column Info:")
        print(pd.Series(self.training_target_scaled).info())
        print("\nTesting Target Column Info:")
        print(pd.Series(self.testing_target_scaled).info())