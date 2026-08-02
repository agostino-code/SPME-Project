import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, arff_path, target_column_name='Effort', test_indices=None, train_indices=None, use_knn_imputation=False, verbose=False):
        if str(arff_path).endswith('.csv'):
            data = pd.read_csv(arff_path)
        else:
            from scipy.io import arff
            raw_data, meta = arff.loadarff(arff_path)
            data = pd.DataFrame(raw_data)
            for col in data.select_dtypes([object]).columns:
                data[col] = data[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        ds_name = str(arff_path).lower()
        
        # Keep only numeric features according to paper
        if 'albrecht' in ds_name:
            pass
        elif 'china' in ds_name:
            drop_cols = ['ID', 'Duration', 'N_effort', 'Resource', 'Dev.Type']
            actual_drops = [c for c in drop_cols if c in data.columns]
            if verbose and len(actual_drops) > 0: print(f"Dropped from China: {actual_drops}", flush=True)
            data = data.drop(columns=actual_drops)
        elif 'desharnais' in ds_name:
            data = data.replace({-1: np.nan, "-1": np.nan, -1.0: np.nan})
            drop_cols = ['id', 'Project', 'YearEnd', 'Language', 'AdjFactor']
            actual_drops = [c for c in drop_cols if c in data.columns]
            if verbose and len(actual_drops) > 0: print(f"Dropped from Desharnais: {actual_drops}", flush=True)
            data = data.drop(columns=actual_drops)
        elif 'cocomo' in ds_name:
            if 'num' in data.columns:
                if verbose: print(f"Dropped from COCOMO: ['num']", flush=True)
                data = data.drop(columns=['num'])
            drop_cols = ['Type', 'DEV']
            actual_drops = [c for c in drop_cols if c in data.columns]
            if verbose and len(actual_drops) > 0: print(f"Dropped from COCOMO: {actual_drops}", flush=True)
            data = data.drop(columns=actual_drops)
        elif 'maxwell' in ds_name:
            drop_cols = ['App', 'Har', 'Dba', 'Ifc', 'Source', 'Telonuse']
            actual_drops = [c for c in drop_cols if c in data.columns]
            if verbose and len(actual_drops) > 0: print(f"Dropped from Maxwell: {actual_drops}", flush=True)
            data = data.drop(columns=actual_drops)

        # Label encode non-numeric before selecting numeric
        for col in data.select_dtypes(['object', 'category', 'string']).columns:
            data[col] = data[col].astype('category').cat.codes
        
        # Drop non-numeric completely just in case (should be empty now)
        data = data.select_dtypes(include=['number'])

        if use_knn_imputation:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=min(5, len(data) - 1))
            cols = data.columns
            data = pd.DataFrame(imputer.fit_transform(data), columns=cols)
        else:
            # Drop rows with missing values
            data = data.dropna().reset_index(drop=True)

        target = data[target_column_name].astype(float)
        features = data.drop(columns=[target_column_name])
        
        self.data = features # All data features

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

        # Min-Max Scaling (Fit on Train only)
        self.feature_min = training_features.min(axis=0)
        self.feature_max = training_features.max(axis=0)
        feature_range = self.feature_max - self.feature_min
        feature_range[feature_range == 0] = 1.0

        self.training_data = ((training_features - self.feature_min) / feature_range).values.T
        self.testing_data = ((testing_features - self.feature_min) / feature_range).values.T

        # Target Scaling (no log transform)
        self.target_min = float(training_target.min())
        self.target_max = float(training_target.max())
        self.target_range = self.target_max - self.target_min if self.target_max != self.target_min else 1.0

        self.training_target_scaled = ((training_target - self.target_min) / self.target_range)
        self.testing_target_scaled = ((testing_target - self.target_min) / self.target_range)

        self.training_features_df = training_features
        self.testing_features_df = testing_features
        self.target_range_val = self.target_range
        self.target_min_val = self.target_min
        self.training_target_unscaled = training_target
        self.testing_target_unscaled = testing_target

    def inverse_transform_target(self, y_scaled):
        y_scaled = np.asarray(y_scaled, dtype=float)
        y_unscaled = (y_scaled * self.target_range_val) + self.target_min_val
        return y_unscaled