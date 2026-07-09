import sys
import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), 'fuzzy-DCNN implementation'))

from dataset import Dataset
from fuzzy import FuzzyCMeans
from DCNN import EffortEstimator

def train_and_save():
    print("Loading dataset...")
    dataset_path = 'Software-estimation-datasets/china.arff'
    target_column = 'Effort'
    dataset = Dataset(dataset_path, target_column_name=target_column)

    print("Training Fuzzy C-Means...")
    n_clusters = 5
    fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=n_clusters)
    mu_train = fuzzy_cmeans.fit_transform()

    print("Training DCNN...")
    n_features = mu_train.shape[1]
    effort_estimator = EffortEstimator(
        n_features, n_clusters, 
        lr=1e-3, 
        batch_size=16, 
        epochs=150, 
        weight_decay=1e-3,
        target_min=dataset.target_min,
        target_range=dataset.target_range,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    effort_estimator.fit(mu_train, dataset.training_target_scaled, verbose=True)

    print("Evaluating DCNN...")
    mu_test = fuzzy_cmeans.transform(dataset.testing_data)
    dcnn_mae, dcnn_rmse, _ = effort_estimator.evaluate(
        mu_test,
        dataset.testing_target_scaled,
        inverse_transform=dataset.inverse_transform_target
    )
    print(f"DCNN Test MAE: {dcnn_mae:.2f}, RMSE: {dcnn_rmse:.2f}")

    print("Training Linear Regression...")
    y_train = dataset.inverse_transform_target(dataset.training_target_scaled)
    y_test = dataset.inverse_transform_target(dataset.testing_target_scaled)

    feature_min = dataset.feature_min.values.reshape(-1, 1)
    feature_range = (dataset.feature_max - dataset.feature_min).values.reshape(-1, 1).copy()
    feature_range[feature_range == 0] = 1.0

    X_train_unscaled = (dataset.training_data * feature_range) + feature_min
    X_test_unscaled = (dataset.testing_data * feature_range) + feature_min

    df_train = pd.DataFrame(X_train_unscaled.T, columns=dataset.data.columns)
    df_test = pd.DataFrame(X_test_unscaled.T, columns=dataset.data.columns)

    df_train['HLFPA'] = 7 * df_train['File'] + 5 * df_train['Interface'] + 4 * df_train['Input'] + 5 * df_train['Output'] + 4 * df_train['Enquiry']
    df_train['IFPA'] = 35 * df_train['File'] + 15 * df_train['Interface']

    df_test['HLFPA'] = 7 * df_test['File'] + 5 * df_test['Interface'] + 4 * df_test['Input'] + 5 * df_test['Output'] + 4 * df_test['Enquiry']
    df_test['IFPA'] = 35 * df_test['File'] + 15 * df_test['Interface']

    lin_reg = LinearRegression()
    lin_reg.fit(df_train[['HLFPA', 'IFPA']], y_train)

    lin_preds = lin_reg.predict(df_test[['HLFPA', 'IFPA']])
    lin_mae = mean_absolute_error(y_test, lin_preds)
    lin_rmse = np.sqrt(mean_squared_error(y_test, lin_preds))

    print(f"Linear Regression Test MAE: {lin_mae:.2f}, RMSE: {lin_rmse:.2f}")

    print("Saving models...")
    os.makedirs("models", exist_ok=True)
    
    torch.save(effort_estimator.model.state_dict(), "models/dcnn.pth")
    
    joblib.dump({
        'n_features': n_features,
        'n_clusters': n_clusters,
        'centroids': fuzzy_cmeans.centroids,
        'deltas': fuzzy_cmeans.deltas,
        'feature_min': dataset.feature_min,
        'feature_max': dataset.feature_max,
        'target_min': dataset.target_min,
        'target_range': dataset.target_range,
        'lin_reg': lin_reg,
        'columns': dataset.data.columns,
        'metrics': {
            'dcnn_mae': float(dcnn_mae),
            'dcnn_rmse': float(dcnn_rmse),
            'lin_mae': float(lin_mae),
            'lin_rmse': float(lin_rmse)
        }
    }, "models/models-meta.joblib")
    
    print("Training complete and models saved.")

if __name__ == "__main__":
    train_and_save()
