import sys
import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), 'fuzzy-DCNN-implementation'))

from dataset import Dataset
from fuzzy import FuzzyCMeans
from DCNN import EffortEstimator

def train_and_save():
    print("Loading dataset...")
    dataset_path = 'Software-estimation-datasets/china.arff'
    target_column = 'Effort'
    dataset = Dataset(dataset_path, target_column_name=target_column, use_log_target=True)

    print("Training Fuzzy C-Means...")
    n_clusters = 5
    fuzzy_cmeans = FuzzyCMeans(dataset, num_clusters=n_clusters, m=2.0)
    mu_train = fuzzy_cmeans.fit_transform()

    print("Training DCNN Regressor with Huber Loss...")
    n_features = mu_train.shape[1]
    target_mean_scaled = float(dataset.training_target_scaled.mean())

    num_filters = 32
    dropout_rate = 0.1
    loss_type = 'huber'
    huber_delta = 0.01
    lr = 1e-3
    epochs = 200
    batch_size = 16
    weight_decay = 0.0
    activation_type = 'linear_clip'

    effort_estimator = EffortEstimator(
        m_features=n_features,
        k_clusters=n_clusters,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        target_min=dataset.target_min,
        target_range=dataset.target_range,
        target_mean=target_mean_scaled,
        num_filters=num_filters,
        dropout_rate=dropout_rate,
        loss_type=loss_type,
        huber_delta=huber_delta,
        activation_type=activation_type,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    effort_estimator.fit(mu_train, dataset.training_target_scaled.values)

    print("Evaluating DCNN...")
    mu_test = fuzzy_cmeans.transform(dataset.testing_data)
    pred_scaled = effort_estimator.predict(mu_test)
    pred_unscaled = dataset.inverse_transform_target(pred_scaled)
    y_test_unscaled = dataset.testing_target_unscaled.values

    dcnn_mae = float(mean_absolute_error(y_test_unscaled, pred_unscaled))
    dcnn_rmse = float(np.sqrt(mean_squared_error(y_test_unscaled, pred_unscaled)))
    abs_errors = np.abs(y_test_unscaled - pred_unscaled)
    mre = abs_errors / np.maximum(y_test_unscaled, 1e-6)
    dcnn_mmre = float(np.mean(mre))
    dcnn_pred25 = float(np.mean(mre <= 0.25) * 100.0)

    print(f"DCNN Test MAE: {dcnn_mae:.2f}, RMSE: {dcnn_rmse:.2f}, MMRE: {dcnn_mmre:.4f}, Pred(25): {dcnn_pred25:.2f}%")

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

    for df in [df_train, df_test]:
        for col in ['File', 'Interface', 'Input', 'Output', 'Enquiry']:
            if col not in df.columns:
                df[col] = 0.0

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
        'target_mean': target_mean_scaled,
        'num_filters': num_filters,
        'dropout_rate': dropout_rate,
        'loss_type': loss_type,
        'huber_delta': huber_delta,
        'activation_type': activation_type,
        'use_log_target': dataset.use_log_target,
        'lin_reg': lin_reg,
        'columns': dataset.data.columns,
        'metrics': {
            'dcnn_mae': float(dcnn_mae),
            'dcnn_rmse': float(dcnn_rmse),
            'dcnn_mmre': float(dcnn_mmre),
            'dcnn_pred25': float(dcnn_pred25),
            'lin_mae': float(lin_mae),
            'lin_rmse': float(lin_rmse)
        }
    }, "models/models-meta.joblib")

    print("Training complete and models saved.")

if __name__ == "__main__":
    train_and_save()

