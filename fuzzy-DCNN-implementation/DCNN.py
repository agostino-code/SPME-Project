import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def safe_pool(x):
    h, w = x.shape[2], x.shape[3]
    kh = min(2, h)
    kw = min(2, w)
    return torch.nn.functional.max_pool2d(x, kernel_size=(kh, kw), stride=1, padding=0)

class FDCNN(nn.Module):
    def __init__(self, m_features, k_clusters, target_mean=0.0, num_filters=16, dropout_rate=0.2):
        super(FDCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.safe_pool = safe_pool
        
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
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(32, 16)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(16, 1)
        
        for m in [self.conv1, self.conv2, self.fc1, self.fc2]:
            init_weights(m)
                
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
        return x

class FDCNN_SeqAsym(nn.Module):
    def __init__(self, m_features, k_clusters, target_mean=0.0, num_filters=16, dropout_rate=0.2):
        super(FDCNN_SeqAsym, self).__init__()
        
        # PROMPT 1: Sequential Asymmetric
        # 1. kernel (1, k)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(1, k_clusters), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Calculate height after conv1 and safe_pool
        # safe_pool uses kernel (min(2, h), min(2, w)) and stride 1, padding 0
        h_after_pool1 = m_features - min(2, m_features) + 1
        
        # 2. kernel (m, 1) -> updated to match height after pool
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(h_after_pool1, 1), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.safe_pool = safe_pool
        
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
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(32, 16)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(16, 1)
        
        for m in [self.conv1, self.conv2, self.fc1, self.fc2]:
            init_weights(m)
                
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
        return x


class FDCNN_ParAsym(nn.Module):
    def __init__(self, m_features, k_clusters, target_mean=0.0, num_filters=16, dropout_rate=0.2):
        super(FDCNN_ParAsym, self).__init__()
        
        # PROMPT 1: Parallel Asymmetric
        self.conv1_A = nn.Conv2d(in_channels=1, out_channels=max(1, num_filters // 2), kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.relu1_A = nn.ReLU()
        
        self.conv1_B = nn.Conv2d(in_channels=1, out_channels=max(1, num_filters // 2), kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.relu1_B = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.safe_pool = safe_pool
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, m_features, k_clusters)
            xA = self.conv1_A(dummy_input)
            xA = self.relu1_A(xA)
            xA = self.safe_pool(xA)
            
            xB = self.conv1_B(dummy_input)
            xB = self.relu1_B(xB)
            xB = self.safe_pool(xB)
            
            x = torch.cat([xA, xB], dim=1)
            flattened_size = x.view(1, -1).shape[1]
            
        self.fc1 = nn.Linear(flattened_size, 32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(32, 16)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(16, 1)
        
        for m in [self.conv1_A, self.conv1_B, self.fc1, self.fc2]:
            init_weights(m)
                
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out.bias, target_mean)

    def forward(self, x):
        xA = self.conv1_A(x)
        xA = self.relu1_A(xA)
        xA = self.safe_pool(xA)
        
        xB = self.conv1_B(x)
        xB = self.relu1_B(xB)
        xB = self.safe_pool(xB)
        
        x = torch.cat([xA, xB], dim=1)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.out(x)
        return x

class EffortEstimator:
    def __init__(self, m_features, k_clusters, lr=0.001, epochs=150, batch_size=16, 
                 target_min=0.0, target_range=1.0, target_mean=0.0, device='cpu', 
                 model_type='FDCNN', num_filters=16, dropout_rate=0.2,
                 loss_type='rmse', huber_delta=0.1):
        self.m_features = m_features
        self.k_clusters = k_clusters
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        if model_type == 'FDCNN_SeqAsym':
            model_class = FDCNN_SeqAsym
        elif model_type == 'FDCNN_ParAsym':
            model_class = FDCNN_ParAsym
        else:
            model_class = FDCNN
            
        self.model = model_class(m_features, k_clusters, target_mean=target_mean, 
                                 num_filters=num_filters, dropout_rate=dropout_rate).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        class RMSELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
            def forward(self,yhat,y):
                return torch.sqrt(self.mse(yhat,y) + 1e-6)

        if loss_type == 'huber':
            # Huber (Smooth-L1): quadratica per errori piccoli, lineare (non esplode)
            # per errori grandi. delta è nella scala del target scalato [0, 1], quindi
            # valori tipici da esplorare sono piccoli (es. 0.05 - 0.5).
            self.criterion = nn.HuberLoss(delta=huber_delta)
        else:
            self.criterion = RMSELoss()

        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.target_min = target_min
        self.target_range = target_range
        
    def fit(self, X_train, y_train):
        self.model.train()
        X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(self.device)
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
        # Effort non può essere negativo: il clipping avviene solo qui, in inferenza,
        # non nel forward pass di training (dove terrebbe il gradiente sempre lineare
        # e libero di fluire, evitando sia il collasso della ReLU sia il bias della Softplus).
        preds = np.maximum(preds, 0.0)
        return preds