import numpy as np
import pandas as pd

class Dataset:
    last_csv = None
    last_data = None
    def __init__(self, arff_path, target_column_name='Effort', test_indices=None):
        if arff_path != Dataset.last_csv or Dataset.last_data is None:
            Dataset.last_csv = arff_path
            if arff_path.endswith('.csv'):
                Dataset.last_data = pd.read_csv(arff_path)
            else:
                from scipy.io import arff
                data, meta = arff.loadarff(arff_path)
                Dataset.last_data = pd.DataFrame(data)
                for col in Dataset.last_data.select_dtypes([object]).columns:
                    Dataset.last_data[col] = Dataset.last_data[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
        data = Dataset.last_data.copy()

        # Handle missing values (e.g. Desharnais missing records)
        data = data.dropna().reset_index(drop=True)

        target = data[target_column_name].astype(float)

        # Z-score Outlier Filtering on Effort (Azzeh et al. 2024 Section 5.1)
        z_scores = np.abs((target - target.mean()) / (target.std() + 1e-8))
        clean_mask = z_scores <= 2.5
        data = data[clean_mask].reset_index(drop=True)
        target = target[clean_mask].reset_index(drop=True)

        # Dataset-specific column drops for nominal/categorical attributes
        drop_columns = [
            'ID', 'Duration', 'N_effort', 'Dev.Type', 'Resource', 'num', 'dev_mode',
            'Language', 'Dev.Env', 'AdjFactor', 'App', 'Har', 'Dba', 'Ifc', 'Source', 'Telonuse'
        ]
        features = data.drop(columns=[target_column_name]).select_dtypes(include=['number'])
        features = features.drop(columns=[c for c in drop_columns if c in features.columns])

        # Top-K Feature Selection via Mutual Information
        self.top_k_features = min(6, features.shape[1])
        if self.top_k_features < features.shape[1]:
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(features.values, np.log1p(target.values), random_state=42)
            top_cols = features.columns[np.argsort(mi)[::-1][:self.top_k_features]]
            features = features[top_cols]

        if test_indices is None:
            training_features = features.sample(frac=0.7, random_state=42)
            testing_features  = features.drop(training_features.index)
        else:
            testing_features = features.iloc[test_indices]
            training_features = features.drop(features.index[test_indices])

        training_target = target.loc[training_features.index].astype(float)
        testing_target = target.loc[testing_features.index].astype(float)

        self.feature_min = training_features.min(axis=0)
        self.feature_max = training_features.max(axis=0)
        feature_range = self.feature_max - self.feature_min
        feature_range[feature_range == 0] = 1.0

        self.training_data = ((training_features - self.feature_min) / feature_range).values.T
        self.testing_data = ((testing_features - self.feature_min) / feature_range).values.T

        # Adaptive Log-Transform: Use log-transform for large target datasets (China, Desharnais, Maxwell, COCOMO), linear for small scale (Albrecht)
        self.use_log_target = True if float(training_target.max()) > 200.0 else False
        if self.use_log_target:
            training_target = np.log1p(training_target)
            testing_target = np.log1p(testing_target)

        self.target_min = float(training_target.min())
        self.target_max = float(training_target.max())
        self.target_range = self.target_max - self.target_min if self.target_max != self.target_min else 1.0

        self.training_target_scaled = ((training_target - self.target_min) / self.target_range)
        self.testing_target_scaled = ((testing_target - self.target_min) / self.target_range)

        self.data = features

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