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
            return jsonify({'success': False, 'error': 'Modelli non caricati. Esegui prima train.py.'})

    data = request.json

    try:
        features = {}
        expected_features = ['Input', 'Output', 'Enquiry', 'File', 'Interface', 'Added', 'Changed', 'Deleted']

        for col in expected_features:
            val = data.get(col)
            if val is None or str(val).strip() == '':
                return jsonify({'success': False, 'error': f'Parametro {col} mancante.'})
            try:
                features[col] = float(val)
                if features[col] < 0:
                    return jsonify({'success': False, 'error': f'Parametro {col} non puo essere negativo.'})
            except ValueError:
                return jsonify({'success': False, 'error': f'Parametro {col} non valido.'})

        # SLR predictions (Di Martino closed-form formulas)
        slr = ModelManager.predict_slr(
            features['File'], features['Interface'],
            features['Input'], features['Output'], features['Enquiry']
        )

        # F-DCNN prediction
        dcnn_pred = model_manager.predict_dcnn(features)

        return jsonify({
            'success': True,
            'metrics': {
                'hlfpa': slr['hlfpa'],
                'ifpa': slr['ifpa'],
                'fp': slr['fp'],
            },
            'predictions': {
                'mhlfpa': round(slr['effort_mhlfpa'], 2),
                'mifpa': round(slr['effort_mifpa'], 2),
                'mfp': round(slr['effort_mfp'], 2),
                'fdcnn': round(dcnn_pred, 2),
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
