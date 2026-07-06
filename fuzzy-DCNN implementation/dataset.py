import numpy as np
import pandas as pd

class Dataset:
    last_csv = None
    last_data = None
    def __init__(self, csv, target_column_name, LOOCV=None):
        if csv != Dataset.last_csv or Dataset.last_data is None:
            Dataset.last_csv = csv
            Dataset.last_data = pd.read_csv(csv)
        data = Dataset.last_data.copy()

        # Remove rows with -1 in 'TeamExp' or 'ManagerExp' if those columns exist
        if {'TeamExp', 'ManagerExp'}.issubset(data.columns):
            data = data[(data['TeamExp'] != -1) & (data['ManagerExp'] != -1)].copy().reset_index(drop=True)

        target = data[target_column_name].astype(float)
        features = data.drop(columns=[target_column_name]).select_dtypes(include=['number'])

        # Drop irrelevant columns if they exist
        drop_columns = [c for c in ['id', 'Project', 'Language', 'AdjFactor', 'Adjustment', 'YearEnd'] if c in features.columns]
        if drop_columns:
            features = features.drop(columns=drop_columns)

        if LOOCV == None:
            training_features = features.sample(frac=0.7, random_state=42)
            testing_features  = features.drop(training_features.index)
        else:
            training_features = features.drop(LOOCV)
            testing_features  = features.loc[[LOOCV]]

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