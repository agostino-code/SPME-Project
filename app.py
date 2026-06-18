import os
import sys
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

# Add current path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from estimation_model import FuzzyDCNNRegressor

app = Flask(__name__)

DATASET_DIR = r"D:\SPME-Project\Software-estimation-datasets"

# In-memory storage for the trained model and dataset details
trained_models = {}

def load_arff(filepath):
    attributes = []
    data_started = False
    data_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@relation'):
                continue
            if line.lower().startswith('@attribute'):
                parts = line.split()
                name = parts[1].strip("'\"")
                attributes.append(name)
                continue
            if line.lower().startswith('@data'):
                data_started = True
                continue
            if data_started:
                data_lines.append(line)
                
    rows = []
    for line in data_lines:
        vals = []
        for v in line.split(','):
            v = v.strip()
            if v == '?' or v == '':
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(v)
        rows.append(vals)
        
    df = pd.DataFrame(rows, columns=attributes)
    df.dropna(inplace=True)
    return df

def load_dataset(filepath):
    if filepath.endswith('.arff'):
        return load_arff(filepath)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if ';' in first_line:
            df = pd.read_csv(filepath, sep=';', na_values=['?'])
        else:
            df = pd.read_csv(filepath, na_values=['?'])
        df.columns = [c.strip().strip("'\"") for c in df.columns]
        df.dropna(inplace=True)
        return df

def detect_target_and_features(df):
    col_lower = [c.lower() for c in df.columns]
    
    # Identify target column
    target_candidates = ['effort', 'actual', 'act_effort', 'dev.eff.hrs.', 'actual.effort', 'mm', 'actual_effort', 'hh_cfp', 'hourstocount', 'totallaborcost']
    target_col = None
    for cand in target_candidates:
        if cand in col_lower:
            idx = col_lower.index(cand)
            target_col = df.columns[idx]
            break
            
    if target_col is None:
        for c in df.columns:
            if 'effort' in c.lower() or 'actual' in c.lower():
                target_col = c
                break
        if target_col is None:
            target_col = df.columns[-1]
            
    drop_candidates = ['id', 'project', 'client.code', 'num', 'recordnumber', 'projectname', 'project_no', 'projct', 'lneff', 'lnsize', 'n_effort']
    drop_cols = [target_col]
    for c in df.columns:
        if c.lower() in drop_candidates or 'name' in c.lower() or 'code' in c.lower() or 'date' in c.lower():
            drop_cols.append(c)
            
    feature_cols = []
    for c in df.columns:
        if c not in drop_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)
                
    return target_col, feature_cols

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    try:
        files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.csv', '.arff'))]
        return jsonify({'datasets': sorted(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_dataset', methods=['GET'])
def get_dataset_details():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename is required'}), 400
        
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        df = load_dataset(filepath)
        target_col, feature_cols = detect_target_and_features(df)
        
        # Get dataset description (first few rows and columns)
        summary = {
            'rows': int(df.shape[0]),
            'cols': int(df.shape[1]),
            'target': target_col,
            'features': feature_cols,
            'preview': df.head(5).to_dict(orient='records')
        }
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.get_json() or {}
    filename = data.get('filename')
    n_clusters_input = data.get('n_clusters', 'auto')
    epochs = int(data.get('epochs', 200))
    lr = float(data.get('lr', 0.005))
    batch_size = int(data.get('batch_size', 16))
    
    if not filename:
        return jsonify({'error': 'Filename is required'}), 400
        
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        df = load_dataset(filepath)
        target_col, feature_cols = detect_target_and_features(df)
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        n_samples = len(y)
        
        # Parse clusters
        if n_clusters_input != 'auto':
            try:
                n_clusters = int(n_clusters_input)
            except ValueError:
                n_clusters = 'auto'
        else:
            n_clusters = 'auto'
            
        # Select cross-validation scheme
        # If dataset is small, use LOO-CV. If larger, use 5-Fold CV.
        if n_samples <= 100:
            cv = LeaveOneOut()
            cv_mode = "Leave-One-Out CV"
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_mode = "5-Fold Cross Validation"
            
        cv_preds = np.zeros(n_samples)
        
        # Run Cross-Validation
        for train_idx, test_idx in cv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            fold_reg = FuzzyDCNNRegressor(
                n_clusters=n_clusters, 
                epochs=epochs, 
                lr=lr, 
                batch_size=batch_size, 
                random_state=42
            )
            fold_reg.fit(X_tr, y_tr)
            cv_preds[test_idx] = fold_reg.predict(X_te)
            
        # Compute metrics based on cross-validation predictions
        mae = float(mean_absolute_error(y, cv_preds))
        mar = float(np.mean(np.abs(y - cv_preds) / np.maximum(y, 1e-8))) * 100.0
        mdape = float(np.median(np.abs((y - cv_preds) / np.maximum(y, 1e-8)))) * 100.0
        r2 = float(r2_score(y, cv_preds))
        
        # Train final model on complete dataset for serving queries
        final_reg = FuzzyDCNNRegressor(
            n_clusters=n_clusters, 
            epochs=epochs, 
            lr=lr, 
            batch_size=batch_size, 
            random_state=42
        )
        final_reg.fit(X, y)
        
        # Save model and columns info to memory
        trained_models[filename] = {
            'regressor': final_reg,
            'features': feature_cols,
            'target': target_col
        }
        
        actual_k = final_reg.centroids.shape[0]
        
        # Generate Actual vs Predicted Plot
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        # Apply style adjustments for a modern dark-palette theme compatibility
        ax.set_facecolor('#1e1e24')
        fig.patch.set_facecolor('#121214')
        
        ax.scatter(y, cv_preds, color='#6366f1', alpha=0.7, edgecolors='w', s=40, label='Predictions')
        max_val = max(np.max(y), np.max(cv_preds))
        min_val = min(np.min(y), np.min(cv_preds))
        ax.plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', linewidth=2, label='Perfect Estimation')
        
        ax.set_title("Actual vs Predicted Effort", color='#ffffff', fontsize=12, pad=10)
        ax.set_xlabel(f"Actual {target_col}", color='#a1a1aa', fontsize=10)
        ax.set_ylabel(f"Predicted {target_col}", color='#a1a1aa', fontsize=10)
        
        ax.tick_params(colors='#a1a1aa', which='both')
        for spine in ax.spines.values():
            spine.set_color('#3f3f46')
            
        ax.legend(facecolor='#1e1e24', edgecolor='#3f3f46', labelcolor='#ffffff')
        ax.grid(True, linestyle=':', color='#3f3f46', alpha=0.5)
        
        plt.tight_layout()
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'metrics': {
                'mae': mae,
                'mar': mar,
                'mdape': mdape,
                'r2': r2
            },
            'cv_mode': cv_mode,
            'actual_clusters': actual_k,
            'plot': plot_base64
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_effort():
    data = request.get_json() or {}
    filename = data.get('filename')
    inputs = data.get('inputs', {})
    
    if not filename:
        return jsonify({'error': 'Filename is required'}), 400
        
    if filename not in trained_models:
        return jsonify({'error': 'Model not trained yet. Please train first.'}), 400
        
    try:
        model_info = trained_models[filename]
        reg = model_info['regressor']
        features = model_info['features']
        
        # Assemble feature values in correct order
        x_input = []
        for f in features:
            if f not in inputs:
                return jsonify({'error': f'Missing value for feature: {f}'}), 400
            x_input.append(float(inputs[f]))
            
        X_test = np.array([x_input])
        pred = float(reg.predict(X_test)[0])
        
        return jsonify({
            'prediction': pred,
            'target': model_info['target']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
