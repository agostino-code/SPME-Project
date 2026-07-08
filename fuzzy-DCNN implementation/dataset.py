import numpy as np
import pandas as pd

class Dataset:
    last_csv = None
    last_data = None
    def __init__(self, arff_path, target_column_name='Effort', test_indices=None, selected_features=None):
        if arff_path != Dataset.last_csv or Dataset.last_data is None:
            Dataset.last_csv = arff_path
            from scipy.io import arff
            data, meta = arff.loadarff(arff_path)
            Dataset.last_data = pd.DataFrame(data)
            for col in Dataset.last_data.select_dtypes([object]).columns:
                Dataset.last_data[col] = Dataset.last_data[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
        data = Dataset.last_data.copy()
        target = data[target_column_name].astype(float)
        features = data.drop(columns=[target_column_name]).select_dtypes(include=['number'])

        # Calcolo feature Di Martino (dataset China)
        features['HLFPA'] = 7 * features['File'] + 5 * features['Interface'] + 4 * features['Input'] + 5 * features['Output'] + 4 * features['Enquiry']
        features['IFPA'] = 35 * features['File'] + 15 * features['Interface']

        if selected_features is not None:
            features = features[[f for f in selected_features if f in features.columns]]
        else:
            drop_columns = ['ID', 'Duration', 'N_effort']
            features = features.drop(columns=[c for c in drop_columns if c in features.columns])

        if test_indices is None:
            training_features = features.sample(frac=0.7, random_state=42)
            testing_features  = features.drop(training_features.index)
        else:
            testing_features = features.iloc[test_indices]
            training_features = features.drop(features.index[test_indices])

        training_target = np.log1p(target.loc[training_features.index].astype(float))
        testing_target = np.log1p(target.loc[testing_features.index].astype(float))

        self.feature_min = training_features.min(axis=0)
        self.feature_max = training_features.max(axis=0)
        feature_range = self.feature_max - self.feature_min
        feature_range[feature_range == 0] = 1.0

        self.training_data = ((training_features - self.feature_min) / feature_range).values.T
        self.testing_data = ((testing_features - self.feature_min) / feature_range).values.T

        self.target_min = float(training_target.min())
        self.target_max = float(training_target.max())
        self.target_range = self.target_max - self.target_min if self.target_max != self.target_min else 1.0

        self.training_target_scaled = ((training_target - self.target_min) / self.target_range)
        self.testing_target_scaled = ((testing_target - self.target_min) / self.target_range)

        self.data = features

    def inverse_transform_target(self, y_scaled):
        y_scaled = np.asarray(y_scaled, dtype=float)
        # Ripristina da [0, 1] in log-space e torna alla scala originale dell'effort.
        return np.expm1((y_scaled * self.target_range) + self.target_min)

    def print_info(self):
        print("Dataset Info:")
        print(self.data.info())
        print("\nTraining Target Column Info:")
        print(pd.Series(self.training_target_scaled).info())
        print("\nTesting Target Column Info:")
        print(pd.Series(self.testing_target_scaled).info())