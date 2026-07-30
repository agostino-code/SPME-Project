"""Count actual epochs per DCNN model — does early stopping kick in?"""
import numpy as np
from sklearn.model_selection import KFold
from dataset import Dataset
from fuzzy import FuzzyCMeans
from DCNN import EffortEstimator, MuDataset, DataLoader
import copy, torch, time

dataset_path = "../Software-estimation-datasets/albrecht.arff"
target_column = "Effort"

ds = Dataset(dataset_path, target_column_name=target_column,
             test_indices=[0,1,2,3,4])

fuzzy_model = FuzzyCMeans(ds, num_clusters=3, m=2.0)
mu_train = fuzzy_model.fit_transform()

# Patch EffortEstimator.fit to count epochs
original_fit = EffortEstimator.fit

def patched_fit(self, mu_train, y_train, X_train_raw=None, verbose=False, patience=25, val_ratio=0.15):
    y_train_arr = np.asarray(y_train)
    n_samples = len(mu_train)
    
    epoch_counts = []
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
        actual_epochs = 0
        
        for epoch in range(self.epochs):
            actual_epochs += 1
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                loss = self.loss_fn(pred, yb)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
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
        
        epoch_counts.append(actual_epochs)
        stopped = "early" if actual_epochs < self.epochs else "full"
        print(f"  Model {model_idx}: {actual_epochs}/{self.epochs} epochs ({stopped})")
    
    return self

# Test with different settings
for epochs, patience, label in [(100, 15, "inner-fold"), (150, 20, "outer-fold")]:
    print(f"\n--- {label} (epochs={epochs}, patience={patience}) ---")
    est = EffortEstimator(
        n_features=mu_train.shape[1], num_clusters=3,
        lr=1e-3, batch_size=16, epochs=epochs,
        kernels=8, target_min=ds.target_min, target_range=ds.target_range,
        use_log_target=ds.use_log_target
    )
    patched_fit(est, mu_train, ds.training_target_scaled, patience=patience, val_ratio=0.15)

# Also check with n_samples <= 10 (inner fold of inner fold)
print(f"\n--- tiny subset (n=8, no val split) ---")
mu_tiny = mu_train[:8]
y_tiny = np.asarray(ds.training_target_scaled)[:8]
est2 = EffortEstimator(
    n_features=mu_tiny.shape[1], num_clusters=3,
    lr=1e-3, batch_size=16, epochs=100,
    kernels=8, target_min=ds.target_min, target_range=ds.target_range,
    use_log_target=ds.use_log_target
)
patched_fit(est2, mu_tiny, y_tiny, patience=15, val_ratio=0.15)
