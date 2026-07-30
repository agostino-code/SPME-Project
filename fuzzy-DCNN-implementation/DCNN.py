"""
For each sample, compute an m x km x k matrix:

    rows = features,

    columns = cluster membership responses.

Feed that matrix to the CNN regressor.

Output one scalar effort value.
"""

""" The DCNN model
receives the generated matrices from the fuzzy model as
input and endeavors to predict the output variable. Illus-
trated in Fig. 6, the DCNN is composed of two 2D con-
volutional layers, two max pooling layers, and a fully
connected layer. Each convolution layer is comprised of 16
kernels (i.e., filters) with a size of (3,3), and is followed by
Relu layers. This number of kernels has been chosen after
searching for the best number that produces better perfor-
mance among all datasets. The stride size in convolution
and pooling layers is set as (1,1). The fully connected
layers contain Flatten layer and two hidden layers with 32
and 16 neurons each. The activation function for both
hidden layers is Relu function. A dropout layer with
probability 0.2 has been added between hidden layers. The
output layer contains only one neuron with Relu activation
function The DCNN model is trained by utilizing Adam
optimizer and root-mean-squared errors as loss function.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

KERNEL_SIZE = 3
KERNELS = 16
STRIDE = 1
HIDDEN_LAYERS = [32, 16]
DROPOUT_PROB = 0.2
OPTIMIZER = torch.optim.Adam

class MAELoss(nn.Module):
    def __init__(self, target_min=0.0, target_range=1.0, use_log_target=False):
        super().__init__()
        self.t_min = target_min
        self.t_range = target_range
        self.use_log_target = use_log_target

    def forward(self, yhat, y):
        yhat_unscaled = (yhat * self.t_range) + self.t_min
        y_unscaled = (y * self.t_range) + self.t_min
        if self.use_log_target:
            yhat_real = torch.expm1(yhat_unscaled)
            y_real = torch.expm1(y_unscaled)
        else:
            yhat_real = yhat_unscaled
            y_real = y_unscaled
        return torch.mean(torch.abs(yhat_real - y_real))

class HuberLoss(nn.Module):
    def __init__(self, target_min=0.0, target_range=1.0, use_log_target=False, beta=100.0):
        super().__init__()
        self.t_min = target_min
        self.t_range = target_range
        self.use_log_target = use_log_target
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, yhat, y):
        yhat_unscaled = (yhat * self.t_range) + self.t_min
        y_unscaled = (y * self.t_range) + self.t_min
        if self.use_log_target:
            yhat_real = torch.expm1(yhat_unscaled)
            y_real = torch.expm1(y_unscaled)
        else:
            yhat_real = yhat_unscaled
class HybridLoss(nn.Module):
    def __init__(self, target_min=0.0, target_range=1.0, use_log_target=False, alpha=0.5, eps=1e-5):
        super().__init__()
        self.t_min = target_min
        self.t_range = target_range
        self.use_log_target = use_log_target
        self.alpha = alpha
        self.eps = eps

    def forward(self, yhat, y):
        yhat_unscaled = (yhat * self.t_range) + self.t_min
        y_unscaled = (y * self.t_range) + self.t_min
        if self.use_log_target:
            yhat_real = torch.expm1(yhat_unscaled)
            y_real = torch.expm1(y_unscaled)
        else:
            yhat_real = yhat_unscaled
            y_real = y_unscaled

        mae = torch.mean(torch.abs(yhat_real - y_real))
        mre = torch.mean(torch.abs(yhat_real - y_real) / torch.clamp(torch.abs(y_real), min=self.eps))
        y_mean = torch.clamp(torch.mean(torch.abs(y_real)), min=1.0)
        mae_norm = mae / y_mean
        return self.alpha * mae_norm + (1.0 - self.alpha) * mre

class MRELoss(nn.Module):
    def __init__(self, target_min=0.0, target_range=1.0, use_log_target=False, eps=1e-5):
        super().__init__()
        self.t_min = target_min
        self.t_range = target_range
        self.use_log_target = use_log_target
        self.eps = eps

    def forward(self, yhat, y):
        yhat_unscaled = (yhat * self.t_range) + self.t_min
        y_unscaled = (y * self.t_range) + self.t_min
        if self.use_log_target:
            yhat_real = torch.expm1(yhat_unscaled)
            y_real = torch.expm1(y_unscaled)
        else:
            yhat_real = yhat_unscaled
            y_real = y_unscaled

        mre = torch.abs(yhat_real - y_real) / torch.clamp(torch.abs(y_real), min=self.eps)
class MuDataset(Dataset):
    def __init__(self, mu_tensor, targets):
        self.X = torch.tensor(mu_tensor, dtype=torch.float32)
        self.y = torch.tensor(np.asarray(targets), dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x, self.y[idx]

from sklearn.ensemble import GradientBoostingRegressor



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        red_dim = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, red_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(red_dim, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        w = torch.mean(x, dim=(2, 3))
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w.unsqueeze(2).unsqueeze(3)

class DCNNRegressor(nn.Module):
    def __init__(self, n_features, num_clusters, kernels=KERNELS, hidden_layers=HIDDEN_LAYERS, dropout_prob=DROPOUT_PROB):
        super().__init__()

        self.conv1 = nn.Conv2d(1, kernels, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.se1 = SEBlock(kernels)
        self.conv2 = nn.Conv2d(kernels, kernels, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.se2 = SEBlock(kernels)
        self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        self.out = nn.Linear(hidden_layers[1], 1)
        self.out.bias.data.fill_(0.0)

        dummy = torch.zeros(1, 1, n_features, num_clusters)
        with torch.no_grad():
            z = self._forward_features(dummy)
            flat_dim = z.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.se1(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.se2(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.out(x)
        x = torch.clamp_min(x, 0.0)
        return x

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

class EffortEstimator:
    def __init__(self, n_features, num_clusters, lr=1e-3, batch_size=16, epochs=150, weight_decay=1e-3, seed=42, device=None, kernels=16, hidden_layers=HIDDEN_LAYERS, dropout_prob=DROPOUT_PROB, target_min=0.0, target_range=1.0, loss_type='hybrid', alpha=0.5, use_log_target=False, n_ensemble=5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_ensemble = n_ensemble
        self.seed = seed
        self.n_features = n_features
        self.num_clusters = num_clusters
        self.kernels = kernels
        self.hidden_layers = hidden_layers
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay

        if loss_type == 'hybrid':
            self.loss_fn = HybridLoss(target_min=target_min, target_range=target_range, use_log_target=use_log_target, alpha=alpha)
        elif loss_type == 'mre':
            self.loss_fn = MRELoss(target_min=target_min, target_range=target_range, use_log_target=use_log_target)
        elif loss_type == 'mae':
            self.loss_fn = MAELoss(target_min=target_min, target_range=target_range, use_log_target=use_log_target)
        else:
            self.loss_fn = HuberLoss(target_min=target_min, target_range=target_range, use_log_target=use_log_target, beta=100.0)

        self.models = []
        for i in range(n_ensemble):
            torch.manual_seed(seed + i * 100)
            model = DCNNRegressor(n_features, num_clusters, kernels=kernels, hidden_layers=hidden_layers, dropout_prob=dropout_prob).to(self.device)
            self.models.append(model)

        self.model = self.models[0]
        self.meta_learner = None
        self.xgb_model = None
        self.svr_model = None
        self.rf_model = None

    def fit(self, mu_train, y_train, X_train_raw=None, verbose=False, patience=25, val_ratio=0.15):
        y_train_arr = np.asarray(y_train)
        n_samples = len(mu_train)

        for model_idx, model in enumerate(self.models):
            torch.manual_seed(self.seed + model_idx * 100)
            np.random.seed(self.seed + model_idx * 100)

            if val_ratio > 0 and n_samples > 10:
                val_size = max(1, int(n_samples * val_ratio))
                perm = np.random.permutation(n_samples)
                val_idx, train_idx = perm[:val_size], perm[val_size:]

                mu_tr, y_tr = mu_train[train_idx], y_train_arr[train_idx]
                mu_val, y_val = mu_train[val_idx], y_train_arr[val_idx]
            else:
                mu_tr, y_tr = mu_train, y_train_arr
                mu_val, y_val = None, None

            ds_tr = MuDataset(mu_tr, y_tr)
            dl_tr = DataLoader(ds_tr, batch_size=min(self.batch_size, len(ds_tr)), shuffle=True)

            if mu_val is not None:
                ds_val = MuDataset(mu_val, y_val)
                dl_val = DataLoader(ds_val, batch_size=min(self.batch_size, len(ds_val)), shuffle=False)
            else:
                dl_val = None

            optim = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.7)

            best_val_loss = float('inf')
            best_model_weights = None
            patience_counter = 0

            for epoch in range(self.epochs):
                model.train()
                total_tr_loss = 0.0
                for xb, yb in dl_tr:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    loss = self.loss_fn(pred, yb)

                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optim.step()

                    total_tr_loss += loss.item() * xb.size(0)

                scheduler.step()

                if dl_val is not None:
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in dl_val:
                            xb, yb = xb.to(self.device), yb.to(self.device)
                            pred = model(xb)
                            val_loss = self.loss_fn(pred, yb)
                            total_val_loss += val_loss.item() * xb.size(0)

                    avg_val_loss = total_val_loss / len(ds_val)

                    if avg_val_loss < best_val_loss - 1e-4:
                        best_val_loss = avg_val_loss
                        best_model_weights = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break

            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)

        # Fit Heterogeneous Stacking Meta-Learner (DCNN + XGBoost + SVR + Random Forest)
        try:
            dcnn_preds = []
            xb_full = torch.tensor(mu_train, dtype=torch.float32).to(self.device)
            if xb_full.ndim == 3:
                xb_full = xb_full.unsqueeze(1)
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred_m = model(xb_full).cpu().numpy().reshape(-1)
                    dcnn_preds.append(pred_m)

            p_dcnn = np.mean(np.array(dcnn_preds), axis=0)

            if X_train_raw is not None and len(X_train_raw) == n_samples:
                X_tr = np.nan_to_num(np.asarray(X_train_raw, dtype=np.float32))
                self.xgb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=self.seed)
                self.xgb_model.fit(X_tr, y_train_arr)
                p_xgb = self.xgb_model.predict(X_tr)

                self.svr_model = SVR(C=10.0, epsilon=0.01, kernel='rbf')
                self.svr_model.fit(X_tr, y_train_arr)
                p_svr = self.svr_model.predict(X_tr)

                self.rf_model = RandomForestRegressor(n_estimators=100, random_state=self.seed)
                self.rf_model.fit(X_tr, y_train_arr)
                p_rf = self.rf_model.predict(X_tr)

                P_train = np.column_stack([p_dcnn, p_xgb, p_svr, p_rf])
            else:
                P_train = np.column_stack(dcnn_preds)

            if len(P_train) >= 10:
                self.meta_learner = GradientBoostingRegressor(loss='absolute_error', n_estimators=50, max_depth=3, random_state=self.seed)
                self.meta_learner.fit(P_train, y_train_arr)
            else:
                self.meta_learner = None
        except Exception:
            self.meta_learner = None

        return self

    def predict(self, mu, X_test_raw=None):
        x = torch.tensor(mu, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.to(self.device)

        dcnn_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x).cpu().numpy().reshape(-1)
                dcnn_preds.append(pred)

        p_dcnn = np.mean(np.array(dcnn_preds), axis=0)

        if self.meta_learner is not None:
            if self.xgb_model is not None and X_test_raw is not None:
                X_te = np.nan_to_num(np.asarray(X_test_raw, dtype=np.float32))
                if X_te.ndim == 1:
                    X_te = X_te.reshape(1, -1)
                p_xgb = self.xgb_model.predict(X_te)
                p_svr = self.svr_model.predict(X_te)
                p_rf = self.rf_model.predict(X_te)
                P_test = np.column_stack([p_dcnn, p_xgb, p_svr, p_rf])
            else:
                P_test = np.column_stack(dcnn_preds)

            final_pred = self.meta_learner.predict(P_test)
            return np.clip(final_pred, 0.0, None)

        return p_dcnn



    def evaluate(self, mu_test, y_test, X_test_raw=None, inverse_transform=None):
        y_pred = self.predict(mu_test, X_test_raw=X_test_raw)
        y_true = np.asarray(y_test).flatten()


        if inverse_transform is not None:
            y_pred = inverse_transform(y_pred)
            y_true = inverse_transform(y_true)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mdae = float(np.median(np.abs(y_pred - y_true)))

        eps = 1e-5
        mre = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
        mmre = float(np.mean(mre))
        mdmre = float(np.median(mre))
        pred25 = float(np.mean(mre <= 0.25) * 100.0)

        return mae, rmse, mdae, mmre, mdmre, pred25