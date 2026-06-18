import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# --- Fuzzy C-Means (FCM) ---
class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, max_iter=150, tol=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.U = None

    def fit(self, X):
        # X shape: (n_samples, n_features)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        
        # Initialize membership matrix U randomly and normalize rows
        U = rng.random((n_samples, self.n_clusters))
        U = U / np.sum(U, axis=1, keepdims=True)

        for iteration in range(self.max_iter):
            U_prev = U.copy()
            
            # Compute centroids
            Um = U ** self.m
            self.centroids = (Um.T @ X) / np.sum(Um.T, axis=1, keepdims=True)
            
            # Compute distances
            dist = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            dist = np.maximum(dist, 1e-8) # Avoid division by zero
            
            # Update U
            denom = np.sum((dist[:, :, None] / dist[:, None, :]) ** (2.0 / (self.m - 1.0)), axis=2)
            U = 1.0 / np.maximum(denom, 1e-8)
            
            # Check convergence
            if np.linalg.norm(U - U_prev) < self.tol:
                break
                
        # Final update of centroids
        Um = U ** self.m
        self.centroids = (Um.T @ X) / np.sum(Um.T, axis=1, keepdims=True)
        self.U = U
        return self

def xie_beni_index(X, centroids, U):
    n_samples = X.shape[0]
    
    # Numerator: sum_{j=1}^k sum_{i=1}^n U_ij^2 * ||C_j - x_i||^2
    dist_sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    numerator = np.sum((U ** 2) * dist_sq)
    
    # Denominator: n * min_{j != t} ||C_j - C_t||^2
    c_diff_sq = np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(c_diff_sq, np.inf)
    min_c_dist_sq = np.min(c_diff_sq)
    
    denominator = n_samples * np.maximum(min_c_dist_sq, 1e-8)
    return numerator / denominator

def compute_widths(X, centroids, U):
    n_samples, n_features = X.shape
    k_clusters = centroids.shape[0]
    widths = np.zeros((k_clusters, n_features))
    
    U_clipped = np.clip(U, 1e-8, 1.0 - 1e-8)
    denom = -2.0 * np.log(U_clipped)
    denom = np.maximum(denom, 1e-8)
    
    for r in range(k_clusters):
        for j in range(n_features):
            diff_sq = (X[:, j] - centroids[r, j]) ** 2
            val = diff_sq / denom[:, r]
            widths[r, j] = np.mean(np.sqrt(np.maximum(val, 0.0)))
            
    widths = np.maximum(widths, 1e-8)
    return widths

def transform_to_2d(X, centroids, widths):
    # Vectorized computation of Gaussian membership function values
    # X shape: (N, m)
    # centroids shape: (k, m) -> centroids.T shape: (m, k)
    # widths shape: (k, m) -> widths.T shape: (m, k)
    # Resulting shape: (N, m, k)
    diff = X[:, :, None] - centroids.T[None, :, :]
    denom = 2.0 * widths.T[None, :, :]
    val = (diff / denom) ** 2
    return np.exp(-val)


# --- Deep Convolutional Neural Network (DCNN) ---
class DCNNModel(nn.Module):
    def __init__(self, input_shape):
        # input_shape: (H, W) where H = features (m), W = clusters (k)
        super(DCNNModel, self).__init__()
        # Conv1: 1 input channel, 16 output channels, kernel size 3x3, padding 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        # Conv2: 16 input channels, 16 output channels, kernel size 3x3, padding 1
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1)
        
        # Determine flattened feature size via dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            dummy = F.relu(self.conv1(dummy))
            kh1, kw1 = min(2, dummy.shape[2]), min(2, dummy.shape[3])
            if kh1 > 0 and kw1 > 0:
                dummy = F.max_pool2d(dummy, kernel_size=(kh1, kw1), stride=(1, 1))
            dummy = F.relu(self.conv2(dummy))
            kh2, kw2 = min(2, dummy.shape[2]), min(2, dummy.shape[3])
            if kh2 > 0 and kw2 > 0:
                dummy = F.max_pool2d(dummy, kernel_size=(kh2, kw2), stride=(1, 1))
            flat_features = dummy.numel()

        self.fc1 = nn.Linear(flat_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Input shape: (N, 1, H, W)
        x = F.relu(self.conv1(x))
        
        kh1, kw1 = min(2, x.shape[2]), min(2, x.shape[3])
        if kh1 > 0 and kw1 > 0:
            x = F.max_pool2d(x, kernel_size=(kh1, kw1), stride=(1, 1))
            
        x = F.relu(self.conv2(x))
        
        kh2, kw2 = min(2, x.shape[2]), min(2, x.shape[3])
        if kh2 > 0 and kw2 > 0:
            x = F.max_pool2d(x, kernel_size=(kh2, kw2), stride=(1, 1))
            
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc_out(x) # No activation to allow standardized outputs
        return x


# --- Wrapper Model for training and prediction ---
class FuzzyDCNNRegressor:
    def __init__(self, n_clusters='auto', epochs=200, lr=0.005, batch_size=16, random_state=42):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Scaling and fitting parameters
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.fcm = None
        self.centroids = None
        self.widths = None
        self.model = None

    def _determine_k(self, X_scaled):
        n_samples = X_scaled.shape[0]
        max_k = min(10, int(np.sqrt(n_samples)))
        if max_k < 2:
            return 2
            
        best_k = 3
        best_xb = float('inf')
        
        for k in range(2, max_k + 1):
            fcm = FuzzyCMeans(n_clusters=k, random_state=self.random_state)
            fcm.fit(X_scaled)
            xb = xie_beni_index(X_scaled, fcm.centroids, fcm.U)
            if xb < best_xb:
                best_xb = xb
                best_k = k
        return best_k

    def fit(self, X, y):
        # Scale inputs
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Determine number of clusters
        if self.n_clusters == 'auto':
            k = self._determine_k(X_scaled)
        else:
            k = int(self.n_clusters)
            
        # Fit Fuzzy C-Means
        self.fcm = FuzzyCMeans(n_clusters=k, random_state=self.random_state)
        self.fcm.fit(X_scaled)
        self.centroids = self.fcm.centroids
        self.widths = compute_widths(X_scaled, self.centroids, self.fcm.U)
        
        # Transform X to 2D representations (N, m, k)
        X_2d = transform_to_2d(X_scaled, self.centroids, self.widths)
        
        # PyTorch model configuration
        # Add channel dimension: (N, 1, m, k)
        X_tensor = torch.tensor(X_2d, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)
        
        m_features, k_clusters = X_scaled.shape[1], k
        self.model = DCNNModel(input_shape=(m_features, k_clusters))
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        n_samples = X_tensor.shape[0]
        
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start+self.batch_size]
                x_b = X_tensor[batch_idx]
                y_b = y_tensor[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model(x_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.feature_scaler.transform(X)
        X_2d = transform_to_2d(X_scaled, self.centroids, self.widths)
        X_tensor = torch.tensor(X_2d, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            outputs_scaled = self.model(X_tensor).numpy()
            
        # Rescale outputs back to original target space
        outputs = self.target_scaler.inverse_transform(outputs_scaled).flatten()
        # Effort must be non-negative
        outputs = np.maximum(outputs, 0.0)
        return outputs
