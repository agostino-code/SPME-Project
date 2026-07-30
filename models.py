import os
import joblib
import numpy as np
import pandas as pd
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'fuzzy-DCNN-implementation'))
from DCNN import EffortEstimator

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.meta = None
        self.dcnn_model = None
        
    def load(self):
        meta_path = os.path.join(self.models_dir, "models-meta.joblib")
        if not os.path.exists(meta_path):
            return False
            
        self.meta = joblib.load(meta_path)
        
        self.dcnn_model = EffortEstimator(
            self.meta['n_features'], 
            self.meta['n_clusters'], 
            lr=1e-3, batch_size=16, epochs=1, weight_decay=1e-3,
            target_min=self.meta['target_min'],
            target_range=self.meta['target_range'],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dcnn_model.model.load_state_dict(torch.load(os.path.join(self.models_dir, "dcnn.pth")))
        self.dcnn_model.model.eval()
        
        return True

    def predict_dcnn(self, input_features):
        """Backward-compatible: returns prediction float only."""
        if not self.meta:
            return 0.0
            
        sample = np.zeros(len(self.meta['columns']))
        for i, col in enumerate(self.meta['columns']):
            sample[i] = input_features.get(col, 0.0)

        # Guard: all-zero features → zero effort
        if np.all(sample == 0):
            return 0.0
            
        feature_range = self.meta['feature_max'] - self.meta['feature_min']
        feature_range[feature_range == 0] = 1.0
        sample_scaled = (sample - self.meta['feature_min'].values) / feature_range.values
        
        mu = np.zeros((self.meta['n_features'], self.meta['n_clusters']))
        for j in range(self.meta['n_features']):
            for r in range(self.meta['n_clusters']):
                c_rj = self.meta['centroids'][r, j]
                d_rj = self.meta['deltas'][r, j]
                mu[j, r] = np.exp(- ((sample_scaled[j] - c_rj) ** 2) / (2.0 * d_rj**2))
                
        pred_scaled = self.dcnn_model.predict(mu)
        pred = (pred_scaled * self.meta['target_range']) + self.meta['target_min']
        if self.meta.get('use_log_target', False):
            pred = np.expm1(pred)
        return float(pred[0])

    def predict_dcnn_with_mu(self, input_features):
        """Returns a tuple (prediction_float, mu_matrix_as_list_of_lists).
        
        The mu matrix is an NxK numpy array converted to nested lists,
        representing fuzzy membership values for each feature/cluster pair.
        """
        if not self.meta:
            return 0.0, []

        sample = np.zeros(len(self.meta['columns']))
        for i, col in enumerate(self.meta['columns']):
            sample[i] = input_features.get(col, 0.0)

        feature_range = self.meta['feature_max'] - self.meta['feature_min']
        feature_range[feature_range == 0] = 1.0
        sample_scaled = (sample - self.meta['feature_min'].values) / feature_range.values

        mu = np.zeros((self.meta['n_features'], self.meta['n_clusters']))
        for j in range(self.meta['n_features']):
            for r in range(self.meta['n_clusters']):
                c_rj = self.meta['centroids'][r, j]
                d_rj = self.meta['deltas'][r, j]
                mu[j, r] = np.exp(- ((sample_scaled[j] - c_rj) ** 2) / (2.0 * d_rj**2))

        pred_scaled = self.dcnn_model.predict(mu)
        pred = (pred_scaled * self.meta['target_range']) + self.meta['target_min']
        if self.meta.get('use_log_target', False):
            pred = np.expm1(pred)
        return float(pred[0]), mu.tolist()

    @staticmethod
    def predict_slr(file_c, interface_c, input_c, output_c, enquiry_c):
        """Compute SLR-based effort estimates using function point metrics.
        
        Returns a dict with keys: hlfpa, ifpa, fp,
        effort_mhlfpa, effort_mifpa, effort_mfp.
        """
        hlfpa = 7 * file_c + 5 * interface_c + 4 * input_c + 5 * output_c + 4 * enquiry_c
        ifpa = 35 * file_c + 15 * interface_c
        fp = file_c + interface_c + input_c + output_c + enquiry_c

        effort_mhlfpa = 74.9 * (hlfpa ** 0.58) if hlfpa != 0 else 0
        effort_mifpa = 539.99 * (ifpa ** 0.272) if ifpa != 0 else 0
        effort_mfp = 87.42 * (fp ** 0.56) if fp != 0 else 0

        return {
            'hlfpa': hlfpa,
            'ifpa': ifpa,
            'fp': fp,
            'effort_mhlfpa': effort_mhlfpa,
            'effort_mifpa': effort_mifpa,
            'effort_mfp': effort_mfp,
        }

    def predict_linear(self, input_features):
        if not self.meta:
            return 0.0
            
        file_c = float(input_features.get('File', 0))
        interface_c = float(input_features.get('Interface', 0))
        input_c = float(input_features.get('Input', 0))
        output_c = float(input_features.get('Output', 0))
        enquiry_c = float(input_features.get('Enquiry', 0))
        
        hlfpa = 7 * file_c + 5 * interface_c + 4 * input_c + 5 * output_c + 4 * enquiry_c
        ifpa = 35 * file_c + 15 * interface_c
        
        pred = self.meta['lin_reg'].predict(pd.DataFrame({'HLFPA': [hlfpa], 'IFPA': [ifpa]}))
        return max(0.0, float(pred[0]))
