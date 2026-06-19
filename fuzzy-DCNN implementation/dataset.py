import pandas as pd

class Dataset:
    def __init__(self, csv, target_column_name):
        self.data = pd.read_csv(csv)
        
        # set the target column, drop it from the data
        self.target = self.data[target_column_name]
        self.data = self.data.drop(columns=[target_column_name])
        
        # filter out non-numeric columns
        self.data = self.data.select_dtypes(include=['number'])

        # training data is 70% of data, testing data is 30%
        self.training_data = self.data.sample(frac=0.7, random_state=42)
        self.testing_data = self.data.drop(self.training_data.index)

        # ensure that training and testing data have the same columns
        assert set(self.training_data.columns) == set(self.testing_data.columns), "Training and testing data columns mismatch."

        # prepare training and testing for fuzz.cluster.cmeans
        self.training_data = self.training_data.values.T
        self.testing_data = self.testing_data.values.T

    def print_info(self):
        print("Dataset Info:")
        print(self.data.info())
        print("\nTarget Column Info:")
        print(self.target.info())