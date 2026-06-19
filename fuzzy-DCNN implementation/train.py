# training implementation of two-step regressor
# - first step: train a fuzzy classifier (Fuzzy C-Means) to assign membership degrees to each class
# - second step: train a DCNN regressor using the membership degrees as input features

# pipeline:
# a sample from the dataset is fed into the fuzzy classifier, which outputs membership degrees for each class (m x k matrix, where m is the number of attributes, k is the number of clusters)
# the membership degrees are then fed into the DCNN regressor, which outputs a single value representing the predicted effort

from fuzzy import FuzzyCMeans
from dataset import Dataset
from DCNN import DCNN

if __name__ == "__main__":
    dataset = Dataset("../Software-estimation-datasets/Desharnais.csv", target_column_name="Effort")
    fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=9)
    mu = fuzzy_cmeans.fit_transform()
    
    # train the DCNN regressor
