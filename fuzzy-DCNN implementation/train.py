# training implementation of two-step regressor
# - first step: train a fuzzy classifier (Fuzzy C-Means) to assign membership degrees to each class
# - second step: train a DCNN regressor using the membership degrees as input features

# pipeline:
# a sample from the dataset is fed into the fuzzy classifier, which outputs membership degrees for each class (m x k matrix, where m is the number of attributes, k is the number of clusters)
# the membership degrees are then fed into the DCNN regressor, which outputs a single value representing the predicted effort

from fuzzy import FuzzyCMeans
from dataset import Dataset
from DCNN import EffortEstimator

if __name__ == "__main__":
    N_CLUSTERS = 9

    # dataset
    dataset = Dataset("../Software-estimation-datasets/Desharnais.csv", target_column_name="Effort")
    
    # fuzzy C-means clustering
    fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=N_CLUSTERS)
    mu = fuzzy_cmeans.fit_transform()
    
    # DCNN regressor training
    n_features = mu.shape[1]
    effort_estimator = EffortEstimator(n_features, N_CLUSTERS)
    effort_estimator.fit(mu, dataset.training_target.values)
    print ("Training completed.")

    # evaluate the model on the test set
    test_mae, test_rmse = effort_estimator.evaluate(fuzzy_cmeans.transform(dataset.testing_data), dataset.testing_target.values)
    print (f"Test MAE: {test_mae:.6f}")
    print (f"Test RMSE: {test_rmse:.6f}")