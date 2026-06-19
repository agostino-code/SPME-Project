# training implementation of two-step regressor
# - first step: train a fuzzy classifier (Fuzzy C-Means) to assign membership degrees to each class
# - second step: train a DCNN regressor using the membership degrees as input features

# pipeline:
# a sample from the dataset is fed into the fuzzy classifier, which outputs membership degrees for each class (m x k matrix, where m is the number of attributes, k is the number of clusters)
# the membership degrees are then fed into the DCNN regressor, which outputs a single value representing the predicted effort

from fuzzy import FuzzyCMeans
from dataset import Dataset
from DCNN import EffortEstimator

def elbow_XB(dataset, max_clusters=15):
    current_best_XB = float('inf')
    current_best_fuzzy_cmeans = None
    current_best_mu = None
    current_best_cluster_n = 0
    for n_clusters in range(2, max_clusters + 1):
        fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=n_clusters)
        mu = fuzzy_cmeans.fit_transform()
        
        current_XB = fuzzy_cmeans.XB()
        if current_XB < current_best_XB:
            current_best_XB = current_XB
            current_best_fuzzy_cmeans = fuzzy_cmeans
            current_best_mu = mu
            current_best_cluster_n = n_clusters

    assert current_best_fuzzy_cmeans is not None, "No valid fuzzy c-means model found."
    assert current_best_mu is not None, "No valid membership degrees found."
    assert current_best_cluster_n > 0, "No valid number of clusters found."

    return current_best_fuzzy_cmeans, current_best_cluster_n, current_best_mu

def train_and_evaluate(dataset_path, target_column_name):
    # dataset
    dataset = Dataset(dataset_path, target_column_name=target_column_name, LOOCV=current_LOO)
    
    # fuzzy C-means clustering
    fuzzy_cmeans, n_clusters, mu = elbow_XB(dataset, max_clusters=15)
    
    # DCNN regressor training
    n_features = mu.shape[1]
    effort_estimator = EffortEstimator(n_features, n_clusters)
    effort_estimator.fit(mu, dataset.training_target.values)
    print ("Training completed.")

    # evaluate the model on the test set
    test_mae, test_rmse = effort_estimator.evaluate(fuzzy_cmeans.transform(dataset.testing_data), dataset.testing_target.values)
    
    return test_mae, test_rmse

if __name__ == "__main__":
    N_CLUSTERS = 9
    DATASET_PATH = "../Software-estimation-datasets/Desharnais.csv"
    TARGET_COLUMN_NAME = "Effort"

    avg_test_mae = 0
    avg_test_rmse = 0
    min_test_mae = float('inf')
    max_test_mae = float('-inf')

    for current_LOO in range(len(Dataset(DATASET_PATH, target_column_name=TARGET_COLUMN_NAME).data)):
        test_mae, test_rmse = train_and_evaluate(DATASET_PATH, TARGET_COLUMN_NAME)
        avg_test_mae += test_mae
        avg_test_rmse += test_rmse
        min_test_mae = min(min_test_mae, test_mae)
        max_test_mae = max(max_test_mae, test_mae)

    avg_test_mae /= len(Dataset(DATASET_PATH, target_column_name=TARGET_COLUMN_NAME).data)
    avg_test_rmse /= len(Dataset(DATASET_PATH, target_column_name=TARGET_COLUMN_NAME).data)

    print(f"Average Test MAE: {avg_test_mae:.4f} (expected 1044 on Desharnais)")
    print(f"Average Test RMSE: {avg_test_rmse:.4f}")
    print(f"Minimum Test MAE: {min_test_mae:.4f}")
    print(f"Maximum Test MAE: {max_test_mae:.4f}")

    """
    Average Test MAE: 3017.5022
    Average Test RMSE: 3017.5022
    Minimum Test MAE: 3.8291
    Maximum Test MAE: 20609.1343
    """