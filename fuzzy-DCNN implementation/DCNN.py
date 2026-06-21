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

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + 1e-8)

LOSS_FN = RMSELoss

class MuDataset(Dataset):
    def __init__(self, mu_tensor, targets):
        self.X = torch.tensor(mu_tensor, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x, self.y[idx]


class DCNNRegressor(nn.Module):
    def __init__(self, n_features, num_clusters):
        super().__init__()
        self.conv1 = nn.Conv2d(1, KERNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.conv2 = nn.Conv2d(KERNELS, KERNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_PROB)

        dummy = torch.zeros(1, 1, n_features, num_clusters)
        with torch.no_grad():
            z = self._forward_features(dummy)
            flat_dim = z.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, HIDDEN_LAYERS[0])
        self.fc2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.out = nn.Linear(HIDDEN_LAYERS[1], 1)
        self.out_activation = nn.ReLU()

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.out(x)
        x = self.out_activation(x)
        return x


class EffortEstimator:
    def __init__(self, n_features, num_clusters, lr=1e-3, batch_size=32, epochs=100, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DCNNRegressor(n_features, num_clusters).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = LOSS_FN()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, mu_train, y_train, verbose=True):
        ds = MuDataset(mu_train, y_train)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item() * xb.size(0)

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch+1}/{self.epochs} loss={total_loss / len(ds):.6f}")

        return self

    def predict(self, mu):
        self.model.eval()
        x = torch.tensor(mu, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.to(self.device)

        with torch.no_grad():
            pred = self.model(x).cpu().numpy().reshape(-1)
        return pred
    
    def evaluate(self, mu_test, y_test):
        y_pred = self.predict(mu_test)
        y_true = y_test.flatten()
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        return mae, rmse