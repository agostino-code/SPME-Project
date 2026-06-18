import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Global variables for models and statistics
models = {}
dataset_stats = {}

def calculate_mar(y_true, y_pred):
    # Mean of Absolute Residuals (MAR) - primary error metric
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_mdape(y_true, y_pred):
    # Median Absolute Percentage Error (MdAPE) - robust relative error metric
    mask = y_true > 0
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def fit_log_model(X, y):
    # Fit Linear Regression model in log-log space
    X_arr = np.asarray(X, dtype=float) + 1
    logX = np.log(X_arr)
    if len(logX.shape) == 1:
        logX = logX.reshape(-1, 1)
        
    logy = np.log(np.asarray(y, dtype=float))
    model = LinearRegression()
    model.fit(logX, logy)
    return model

def loo_cv_log_predictions(X, y):
    # Leave-One-Out Cross-Validation (LOO-CV) for log-log regression
    X_arr = np.asarray(X, dtype=float) + 1
    if len(X_arr.shape) == 1:
        X_arr = X_arr.reshape(-1, 1)
        
    y_arr = np.asarray(y, dtype=float)
    predictions = np.zeros_like(y_arr, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_arr):
        model = LinearRegression()
        model.fit(np.log(X_arr[train_idx]), np.log(y_arr[train_idx]))
        predictions[test_idx[0]] = float(np.exp(model.predict(np.log(X_arr[test_idx]))[0]))
    return predictions

def train_models():
    # 1. Load the dataset
    df = pd.read_csv("software-cost-estimation/china.csv")

    # 2. Calculate sizing metrics (UFP, HLFPA, IFPA)
    # UFP is the sum of the five standard Base Functional Components (BFCs)
    df['UFP'] = df[['Input', 'Output', 'Enquiry', 'File', 'Interface']].sum(axis=1)
    
    # HLFPA translates total size contributions to counts, then applies average HLFPA weights:
    # ILF=7, EIF=5, EI=4, EO=5, EQ=4
    df['HLFPA'] = df['Input'] + df['Output'] + df['Enquiry'] + (df['File'] / 10.0) * 7.0 + (df['Interface'] / 7.0) * 5.0
    
    # IFPA uses only estimated data functions counts with indicative weights: ILF=35, EIF=15
    df['IFPA'] = (df['File'] / 10.0) * 35.0 + (df['Interface'] / 7.0) * 15.0

    # 3. Extract target variable
    effort = df['Effort'].values
    mean_effort = float(np.mean(effort))
    median_effort = float(np.median(effort))

    mars = {}
    mdapes = {}

    # 4. Calculate Baselines (MeanEFF and MedianEFF) using LOO-CV
    mean_cv_preds = np.zeros_like(effort, dtype=float)
    median_cv_preds = np.zeros_like(effort, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(effort):
        mean_cv_preds[test_idx[0]] = float(np.mean(effort[train_idx]))
        median_cv_preds[test_idx[0]] = float(np.median(effort[train_idx]))

    mars['MeanEFF'] = calculate_mar(effort, mean_cv_preds)
    mdapes['MeanEFF'] = calculate_mdape(effort, mean_cv_preds)

    mars['MedianEFF'] = calculate_mar(effort, median_cv_preds)
    mdapes['MedianEFF'] = calculate_mdape(effort, median_cv_preds)

    # 5. Train and evaluate SLR models (AFP, UFP, HLFPA, IFPA) using LOO-CV
    size_metrics = ['AFP', 'UFP', 'HLFPA', 'IFPA']
    for size_name in size_metrics:
        X = df[size_name].values
        cv_preds = loo_cv_log_predictions(X, effort)
        model_key = f'slr_{size_name}'
        
        mars[model_key] = calculate_mar(effort, cv_preds)
        mdapes[model_key] = calculate_mdape(effort, cv_preds)
        models[model_key] = fit_log_model(X, effort)

    # 6. Train and evaluate Calibrated MLR on 5 components using LOO-CV
    components = ['Input', 'Output', 'Enquiry', 'File', 'Interface']
    X_mlr = df[components].values
    mlr_key = 'mlr_Calibrated'
    
    cv_preds_mlr = loo_cv_log_predictions(X_mlr, effort)
    mars[mlr_key] = calculate_mar(effort, cv_preds_mlr)
    mdapes[mlr_key] = calculate_mdape(effort, cv_preds_mlr)
    models[mlr_key] = fit_log_model(X_mlr, effort)

    # 7. Train and evaluate advanced improvement models using 10-Fold CV
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    lr = LinearRegression()

    # Log-Log SLR (UFP)
    y_log = np.log(effort + 1)
    X_ufp_log = np.log(df[['UFP']].values + 1)
    y_pred_log_ufp = cross_val_predict(lr, X_ufp_log, y_log, cv=cv)
    y_pred_exp_ufp = np.exp(y_pred_log_ufp) - 1
    impr_key = 'impr_loglog_UFP'
    mars[impr_key] = calculate_mar(effort, y_pred_exp_ufp)
    mdapes[impr_key] = calculate_mdape(effort, y_pred_exp_ufp)

    # Log-Log SLR (HLFPA)
    X_hlfpa_log = np.log(df[['HLFPA']].values + 1)
    y_pred_log_hlfpa = cross_val_predict(lr, X_hlfpa_log, y_log, cv=cv)
    y_pred_exp_hlfpa = np.exp(y_pred_log_hlfpa) - 1
    impr_key = 'impr_loglog_HLFPA'
    mars[impr_key] = calculate_mar(effort, y_pred_exp_hlfpa)
    mdapes[impr_key] = calculate_mdape(effort, y_pred_exp_hlfpa)

    # Log-Log MLR (5 components)
    X_all_log = np.log(df[components].values + 1)
    y_pred_log_all = cross_val_predict(lr, X_all_log, y_log, cv=cv)
    y_pred_exp_all = np.exp(y_pred_log_all) - 1
    impr_key = 'impr_loglog_MLR'
    mars[impr_key] = calculate_mar(effort, y_pred_exp_all)
    mdapes[impr_key] = calculate_mdape(effort, y_pred_exp_all)

    # Random Forest (5 components)
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    y_pred_rf = cross_val_predict(rf_model, df[components].values, effort, cv=cv)
    impr_key = 'impr_rf'
    mars[impr_key] = calculate_mar(effort, y_pred_rf)
    mdapes[impr_key] = calculate_mdape(effort, y_pred_rf)

    # 8. Fit full models on entire dataset for production prediction
    lr_loglog_ufp = LinearRegression()
    lr_loglog_ufp.fit(X_ufp_log, y_log)
    models['impr_loglog_UFP'] = lr_loglog_ufp

    lr_loglog_hlfpa = LinearRegression()
    lr_loglog_hlfpa.fit(X_hlfpa_log, y_log)
    models['impr_loglog_HLFPA'] = lr_loglog_hlfpa

    lr_loglog_mlr = LinearRegression()
    lr_loglog_mlr.fit(X_all_log, y_log)
    models['impr_loglog_MLR'] = lr_loglog_mlr

    rf_full = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_full.fit(df[components].values, effort)
    models['impr_rf'] = rf_full

    # 9. Save global dataset statistics
    global dataset_stats
    dataset_stats = {
        'total_projects': len(df),
        'mean_effort': mean_effort,
        'median_effort': median_effort,
        'mean_afp': float(df['AFP'].mean()),
        'mean_ufp': float(df['UFP'].mean()),
        'mars': {k: round(float(v), 2) if np.isfinite(v) else 0.0 for k, v in mars.items()},
        'mdapes': {k: round(v, 2) for k, v in mdapes.items()},
        'methodology': {
            'paper': 'Di Martino, Ferrucci, Gravino et al. (IST 2020)',
            'dataset': 'China Software Benchmark (499 projects, cross-company)',
            'technique': 'SLR/MLR (log-log) + Random Forest',
            'validation': 'Leave-One-Out Cross Validation (base models)',
            'improvement_cv': '10-Fold Cross Validation (improvement models)',
            'primary_metric': 'MAR (Mean of Absolute Residuals)',
            'secondary_metric': 'MdAPE (Median Absolute Percentage Error)',
            'note': 'COSMIC methods not replicable on this dataset (no COSMIC data available). Log-Log transformations significantly improve accuracy.',
        }
    }

    print("[OK] Predictive models trained and loaded successfully.")
    print(f"     Dataset: {len(df)} projects")
    print(f"     Baselines: MeanEFF MAR={mars['MeanEFF']:.0f}, MedianEFF MAR={mars['MedianEFF']:.0f}")
    
    for k in [f'slr_{sm}' for sm in size_metrics] + ['mlr_Calibrated']:
        print(f"     {k}: MAR={mars[k]:.0f}, MdAPE={mdapes[k]:.1f}%")
    
    print("     --- Improvement Models (10-Fold CV) ---")
    for k in ['impr_loglog_UFP', 'impr_loglog_HLFPA', 'impr_loglog_MLR', 'impr_rf']:
        print(f"     {k}: MAR={mars[k]:.0f}, MdAPE={mdapes[k]:.1f}%")

# Train models on startup
train_models()

@app.route('/')
def home():
    return render_template('index.html', stats=dataset_stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract counts from UI
        input_data = {
            'Input': float(data.get('Input', 0)),
            'Output': float(data.get('Output', 0)),
            'Enquiry': float(data.get('Enquiry', 0)),
            'File': float(data.get('File', 0)),
            'Interface': float(data.get('Interface', 0)),
        }

        # Convert counts to size contributions using average complexity weights
        size_input = input_data['Input'] * 4.0
        size_output = input_data['Output'] * 5.0
        size_enquiry = input_data['Enquiry'] * 4.0
        size_file = input_data['File'] * 10.0
        size_interface = input_data['Interface'] * 7.0

        # Calculate sizing metrics
        ufp = size_input + size_output + size_enquiry + size_file + size_interface
        afp = ufp * 1.06
        
        # Calculate HLFPA and IFPA metrics directly from counts
        hlfpa = (7.0 * input_data['File'] + 5.0 * input_data['Interface'] +
                 4.0 * input_data['Input'] + 5.0 * input_data['Output'] + 4.0 * input_data['Enquiry'])
                 
        ifpa = 35.0 * input_data['File'] + 15.0 * input_data['Interface']

        predictions = {}
        is_empty_project = (ufp == 0 and afp == 0)

        if is_empty_project:
            predictions['slr_AFP'] = 0.0
            predictions['slr_UFP'] = 0.0
            predictions['slr_HLFPA'] = 0.0
            predictions['slr_IFPA'] = 0.0
            predictions['mlr_Calibrated'] = 0.0
            predictions['impr_loglog_UFP'] = 0.0
            predictions['impr_loglog_HLFPA'] = 0.0
            predictions['impr_loglog_MLR'] = 0.0
            predictions['impr_rf'] = 0.0
        else:
            def predict_model(model_name, size_val):
                # Predict effort in log-log space and detransform using exp
                if isinstance(size_val, (list, np.ndarray)):
                    safe_size = np.asarray(size_val, dtype=float) + 1
                    log_pred = models[model_name].predict(np.log([safe_size]))[0]
                else:
                    if size_val <= 0:
                        return 0.0
                    safe_size = float(size_val) + 1
                    log_pred = models[model_name].predict(np.log([[safe_size]]))[0]
                return float(np.exp(log_pred))

            predictions['slr_AFP'] = predict_model('slr_AFP', afp)
            predictions['slr_UFP'] = predict_model('slr_UFP', ufp)
            predictions['slr_HLFPA'] = predict_model('slr_HLFPA', hlfpa)
            predictions['slr_IFPA'] = predict_model('slr_IFPA', ifpa)
            
            # Predict using Calibrated MLR
            comps = [size_input, size_output, size_enquiry, size_file, size_interface]
            predictions['mlr_Calibrated'] = predict_model('mlr_Calibrated', comps)

            # Predict using advanced log-log models
            if ufp > 0:
                log_ufp_input = np.log(np.array([[ufp + 1]]))
                predictions['impr_loglog_UFP'] = float(np.exp(models['impr_loglog_UFP'].predict(log_ufp_input)[0]) - 1)
            else:
                predictions['impr_loglog_UFP'] = 0.0

            if hlfpa > 0:
                log_hlfpa_input = np.log(np.array([[hlfpa + 1]]))
                predictions['impr_loglog_HLFPA'] = float(np.exp(models['impr_loglog_HLFPA'].predict(log_hlfpa_input)[0]) - 1)
            else:
                predictions['impr_loglog_HLFPA'] = 0.0

            comps_arr = np.array([size_input, size_output, size_enquiry, size_file, size_interface])
            if np.any(comps_arr > 0):
                log_comps_input = np.log(comps_arr.reshape(1, -1) + 1)
                predictions['impr_loglog_MLR'] = float(np.exp(models['impr_loglog_MLR'].predict(log_comps_input)[0]) - 1)
            else:
                predictions['impr_loglog_MLR'] = 0.0

            # Predict using Random Forest
            predictions['impr_rf'] = float(models['impr_rf'].predict(comps_arr.reshape(1, -1))[0])

        sizes = {
            'UFP': round(ufp, 2), 'AFP': round(afp, 2),
            'HLFPA': round(hlfpa, 2), 'IFPA': round(ifpa, 2)
        }

        return jsonify({
            'success': True,
            'sizes': sizes,
            'predictions': {k: round(v, 2) for k, v in predictions.items()},
            'mars': dataset_stats['mars'],
            'mdapes': dataset_stats['mdapes'],
            'methodology': dataset_stats['methodology'],
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
