import pandas as pd

class Dataset:
    last_csv = None
    last_data = None
    def __init__(self, csv, target_column_name, LOOCV=None):
        if csv != Dataset.last_csv or Dataset.last_data is None:
            Dataset.last_csv = csv
            Dataset.last_data = pd.read_csv(csv)
        data = Dataset.last_data.copy()

        # Separate target before splitting
        target = data[target_column_name]
        features = data.drop(columns=[target_column_name]).select_dtypes(include=['number'])

        # also drop id and Project (deshernais dataset specific, see other datasets for other columns to drop)
        features = features.drop(columns=['id', 'Project'])

        # Split features and targets together
        if LOOCV == None:
            training_features = features.sample(frac=0.7, random_state=42)
            testing_features  = features.drop(training_features.index)
        else:
            training_features = features.drop(LOOCV)
            testing_features  = features.loc[[LOOCV]]

        self.training_target = target.loc[training_features.index]
        self.testing_target  = target.loc[testing_features.index]

        assert set(training_features.columns) == set(testing_features.columns)

        self.data          = features
        self.training_data = training_features.values.T
        self.testing_data  = testing_features.values.T

    def print_info(self):
        print("Dataset Info:")
        print(self.data.info())
        print("\nTraining Target Column Info:")
        print(self.training_target.info())
        print("\nTesting Target Column Info:")
        print(self.testing_target.info())