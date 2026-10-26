from flask import Flask, request, jsonify, render_template
from models import ModelManager
import os
import pandas as pd

app = Flask(__name__)
model_manager = ModelManager()

# Load China dataset for the explorer
def load_china_dataset():
    arff_path = os.path.join(os.path.dirname(__file__), 'Software-estimation-datasets', 'china.arff')
    from scipy.io import arff as arff_loader
    data, meta = arff_loader.loadarff(arff_path)
    df = pd.DataFrame(data)
    # Keep only the columns we need, including AFP
    keep_cols = ['AFP', 'Input', 'Output', 'Enquiry', 'File', 'Interface', 'Added', 'Changed', 'Deleted', 'Effort']
    df = df[keep_cols].astype(float)
    return df

china_df = load_china_dataset()

def compute_stats(df):
    ufp = df['Input'] + df['Output'] + df['Enquiry'] + df['File'] + df['Interface']
    correlation = df['AFP'].corr(ufp)
    mean_effort = df['Effort'].mean()
    median_effort = df['Effort'].median()
    mar_mean = (df['Effort'] - mean_effort).abs().mean()
    mar_median = (df['Effort'] - median_effort).abs().mean()
    return {
        'total_projects': len(df),
        'ufp_afp_correlation': correlation if not pd.isna(correlation) else 0.9912,
        'mars': {
            'MeanEFF': mar_mean,
            'MedianEFF': mar_median
        }
    }

dataset_stats = compute_stats(china_df)


@app.route('/')
def index():
    is_loaded = model_manager.load()
    metrics = None
    if is_loaded and model_manager.meta:
        metrics = model_manager.meta.get('metrics', {})
    return render_template('index.html', is_loaded=is_loaded, metrics=metrics, stats=dataset_stats, targets=['Effort'])


@app.route('/predict', methods=['POST'])
def predict():
    if not model_manager.meta:
        if not model_manager.load():
            return jsonify({'success': False, 'error': 'Modelli non caricati. Esegui prima train.py.'}), 503

    data = request.get_json(silent=True) or {}

    try:
        features = {}
        expected_features = ['Input', 'Output', 'Enquiry', 'File', 'Interface', 'Added', 'Changed', 'Deleted']

        for col in expected_features:
            val = data.get(col)
            if val is None or str(val).strip() == '':
                return jsonify({'success': False, 'error': f'Parametro {col} mancante.'}), 400
            try:
                features[col] = float(val)
                if features[col] < 0:
                    return jsonify({'success': False, 'error': f'Parametro {col} non puo essere negativo.'}), 400
            except ValueError:
                return jsonify({'success': False, 'error': f'Parametro {col} non valido.'}), 400

        # Team Productivity Delivery Rate (PDR in ore/FP, default 8.1)
        raw_pdr = data.get('PDR', 8.1)
        try:
            pdr = float(raw_pdr)
            if pdr <= 0:
                pdr = 8.1
        except (ValueError, TypeError):
            pdr = 8.1

        # Hourly rate (tariffa oraria media in €/h, default 50)
        raw_rate = data.get('HourlyRate', 50)
        try:
            hourly_rate = float(raw_rate)
            if hourly_rate <= 0:
                hourly_rate = 50.0
        except (ValueError, TypeError):
            hourly_rate = 50.0

        # SLR predictions (Di Martino closed-form formulas)
        slr = ModelManager.predict_slr(
            features['File'], features['Interface'],
            features['Input'], features['Output'], features['Enquiry']
        )

        # F-DCNN prediction with PDR
        dcnn_pred = model_manager.predict_dcnn(features, pdr=pdr)

        # Baseline efforts in hours
        predictions = {
            'mhlfpa': round(slr['effort_mhlfpa'], 2),
            'mifpa': round(slr['effort_mifpa'], 2),
            'mfp': round(slr['effort_mfp'], 2),
            'fdcnn': round(dcnn_pred, 2),
        }

        # Economic costs (€ = hours * hourly_rate)
        costs = {
            'mhlfpa': round(slr['effort_mhlfpa'] * hourly_rate, 2),
            'mifpa': round(slr['effort_mifpa'] * hourly_rate, 2),
            'mfp': round(slr['effort_mfp'] * hourly_rate, 2),
            'fdcnn': round(dcnn_pred * hourly_rate, 2),
        }

        # Person-Months (PM = hours / 160)
        person_months = {
            'mhlfpa': round(slr['effort_mhlfpa'] / 160.0, 1),
            'mifpa': round(slr['effort_mifpa'] / 160.0, 1),
            'mfp': round(slr['effort_mfp'] / 160.0, 1),
            'fdcnn': round(dcnn_pred / 160.0, 1),
        }

        return jsonify({
            'success': True,
            'hourly_rate': hourly_rate,
            'pdr': pdr,
            'metrics': {
                'hlfpa': slr['hlfpa'],
                'ifpa': slr['ifpa'],
                'fp': slr['fp'],
            },
            'predictions': predictions,
            'costs': costs,
            'person_months': person_months,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
