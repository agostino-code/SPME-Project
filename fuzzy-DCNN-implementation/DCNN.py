import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FDCNN(nn.Module):
    def __init__(self, m_features, k_clusters, target_mean=0.0):
        super(FDCNN, self).__init__()
        
        # Input shape: (batch_size, 1, m_features, k_clusters)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # Custom pooling function to prevent dimension collapse
        def safe_pool(x):
            h, w = x.shape[2], x.shape[3]
            kh = min(2, h)
            kw = min(2, w)
            return torch.nn.functional.max_pool2d(x, kernel_size=(kh, kw), stride=1, padding=0)
            
        self.safe_pool = safe_pool
        
        # Calculate flatten size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, m_features, k_clusters)
            x = self.conv1(dummy_input)
            x = self.relu1(x)
            x = self.safe_pool(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.safe_pool(x)
            flattened_size = x.view(1, -1).shape[1]
            
        self.fc1 = nn.Linear(flattened_size, 32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(16, 1)
        self.relu_out = nn.ReLU()
        
        # He/Kaiming Initialization for all Conv and Linear (except out)
        for m in [self.conv1, self.conv2, self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        # Initialize output layer bias to the mean of the training target to prevent dead ReLU
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out.bias, target_mean)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.safe_pool(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.safe_pool(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.out(x)
        x = self.relu_out(x)
        return x

class EffortEstimator:
    def __init__(self, m_features, k_clusters, lr=0.001, epochs=150, batch_size=16, target_min=0.0, target_range=1.0, target_mean=0.0, device='cpu'):
        self.m_features = m_features
        self.k_clusters = k_clusters
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model = FDCNN(m_features, k_clusters, target_mean=target_mean).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # we'll use MSE for training, RMSE is sqrt of MSE
        self.target_min = target_min
        self.target_range = target_range
        
    def fit(self, X_train, y_train):
        self.model.train()
        
        # X_train is (batch_size, m_features, k_clusters)
        X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(self.device) # Add channel dim
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
    def predict(self, X_test):
        self.model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze(1).cpu().numpy()
        return preds