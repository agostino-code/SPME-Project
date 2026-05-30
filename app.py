import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy import stats as scipy_stats

app = Flask(__name__)

# Global dictionary to hold our trained models
models = {}
dataset_stats = {}


# ---------------------------------------------------------------------------
# Evaluation metrics (as prescribed by the paper)
# ---------------------------------------------------------------------------

def calculate_mar(y_true, y_pred):
    """Mean of Absolute Residuals – primary metric used in the paper."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mdape(y_true, y_pred):
    """Median Absolute Percentage Error – supplementary metric."""
    mask = y_true > 0
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Statistical tests (Section 3.5 of the paper)
# ---------------------------------------------------------------------------

def wilcoxon_test(residuals_model, residuals_baseline):
    """
    Wilcoxon signed-rank test (one-tailed) on paired absolute residuals.
    Tests whether model residuals are significantly SMALLER than baseline residuals.
    Returns p-value for the one-sided alternative (model < baseline).
    """
    diff = residuals_baseline - residuals_model
    try:
        stat, p_two_sided = scipy_stats.wilcoxon(diff, alternative='greater')
        return float(p_two_sided)
    except ValueError:
        # All differences are zero (identical predictions)
        return 1.0


def vargha_delaney_a12(group1, group2):
    """
    Vargha-Delaney A12 effect size statistic.
    A12 = P(X1 < X2) + 0.5 * P(X1 == X2)
    Values:
        A12 ≈ 0.5  → negligible
        A12 ≈ 0.56 → small
        A12 ≈ 0.64 → medium
        A12 ≈ 0.71 → large
    Here group1 = model residuals, group2 = baseline residuals.
    We want group1 < group2 (model is better), so A12 > 0.5 means model is better.
    """
    m, n = len(group1), len(group2)
    r = scipy_stats.rankdata(np.concatenate([group1, group2]))
    r1 = r[:m]
    a12 = (np.sum(r1) - m * (m + 1) / 2) / (m * n)
    # Flip: we want P(baseline > model), i.e., baseline residuals are larger
    return float(1 - a12)


def classify_effect_size(a12):
    """Classify Vargha-Delaney A12 into negligible/small/medium/large."""
    a12_adj = max(a12, 1 - a12)  # symmetric
    if a12_adj < 0.56:
        return "negligible"
    elif a12_adj < 0.64:
        return "small"
    elif a12_adj < 0.71:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Model diagnostics (Section 3.2 of the paper)
# ---------------------------------------------------------------------------

def model_diagnostics(X, y):
    """
    Compute full SLR diagnostics in log-log space as described in the paper.
    Returns a dict with: R2, adj_R2, F_stat, sign_F, equation coefficients,
    Shapiro-Wilk test on residuals, Cook's distances, influential observations.
    """
    X_safe = np.maximum(np.asarray(X, dtype=float), np.finfo(float).eps)
    logX = np.log(X_safe).reshape(-1, 1)
    logy = np.log(np.asarray(y, dtype=float))

    n = len(logy)
    model = LinearRegression()
    model.fit(logX, logy)

    logy_pred = model.predict(logX)
    residuals = logy - logy_pred

    # R² and Adjusted R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p = 1  # number of predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    # F-statistic and its p-value (Sign.F)
    if ss_res > 0 and n > p + 1:
        f_stat = (ss_tot - ss_res) / p / (ss_res / (n - p - 1))
        sign_f = float(1 - scipy_stats.f.cdf(f_stat, p, n - p - 1))
    else:
        f_stat = float('inf')
        sign_f = 0.0

    # Shapiro-Wilk test on residuals (normality assumption)
    if n >= 3:
        sw_stat, sw_p = scipy_stats.shapiro(residuals)
    else:
        sw_stat, sw_p = float('nan'), float('nan')

    # Cook's Distance
    hat_matrix_diag = logX.flatten() ** 2 / np.sum(logX.flatten() ** 2) + 1 / n  # leverage approximation
    mse = ss_res / (n - p - 1) if n > p + 1 else ss_res / max(n - 1, 1)
    cooks_d = (residuals ** 2 * hat_matrix_diag) / ((p + 1) * mse * (1 - hat_matrix_diag) ** 2)
    threshold = 4.0 / n
    influential_indices = list(np.where(cooks_d > threshold)[0])

    # Equation coefficients: EFF = exp(a) * Size^b → log(EFF) = a + b*log(Size)
    b_coef = float(model.coef_[0])
    a_coef = float(np.exp(model.intercept_))

    return {
        'r2': round(float(r2), 4),
        'adj_r2': round(float(adj_r2), 4),
        'f_stat': round(float(f_stat), 2),
        'sign_f': round(float(sign_f), 6),
        'equation': f'EFF = {a_coef:.2f} × Size^{b_coef:.3f}',
        'a_coef': round(a_coef, 2),
        'b_coef': round(b_coef, 3),
        'shapiro_w': round(float(sw_stat), 4),
        'shapiro_p': round(float(sw_p), 4),
        'n_influential': len(influential_indices),
        'cooks_threshold': round(threshold, 4),
    }


# ---------------------------------------------------------------------------
# SLR model fitting and LOO-CV (Section 3.2 and 3.5 of the paper)
# ---------------------------------------------------------------------------

def fit_log_slr(X, y):
    """Fit a Simple Linear Regression in log-log space: log(EFF) = a + b*log(Size)."""
    X_safe = np.maximum(np.asarray(X, dtype=float), np.finfo(float).eps)
    logX = np.log(X_safe).reshape(-1, 1)
    logy = np.log(np.asarray(y, dtype=float))
    model = LinearRegression()
    model.fit(logX, logy)
    return model


def loo_cv_log_slr_predictions(X, y):
    """
    Leave-One-Out Cross Validation for log-log SLR.
    For each fold: train on n-1 observations, predict the held-out one.
    Returns predictions in original scale (not log).
    """
    X_arr = np.maximum(np.asarray(X, dtype=float), np.finfo(float).eps)
    y_arr = np.asarray(y, dtype=float)
    predictions = np.zeros_like(y_arr, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_arr):
        model = LinearRegression()
        model.fit(np.log(X_arr[train_idx]).reshape(-1, 1), np.log(y_arr[train_idx]))
        predictions[test_idx[0]] = float(np.exp(model.predict(np.log(X_arr[test_idx]).reshape(-1, 1))[0]))
    return predictions


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_models():
    """
    Replicate the empirical study design from Di Martino et al. (IST 2020)
    on the China dataset (499 projects, cross-company).

    The paper's methodology is applied as follows:
    1. Compute size metrics (UFP, AFP, HLFPA, IFPA)
    2. Fit SLR models in log-log space
    3. Validate with LOO-CV
    4. Evaluate with MAR and MdAPE
    5. Apply Wilcoxon signed-rank tests with Bonferroni correction
    6. Compute Vargha-Delaney A12 effect size
    7. Collect model diagnostics (R², F, Shapiro-Wilk, Cook's distance)
    """
    # Load dataset
    df = pd.read_csv("software-cost-estimation/china.csv")

    # Precompute custom sizing metrics (Section 2.2 and 2.3 of the paper)
    # UFP = Unadjusted Function Points (sum of the 5 BFC types)
    df['UFP'] = df[['Input', 'Output', 'Enquiry', 'File', 'Interface']].sum(axis=1)

    # HLFPA = High-Level FPA: weighted sum with average complexity weights
    # ILF(File)=7, EIF(Interface)=5, EI(Input)=4, EO(Output)=5, EQ(Enquiry)=4
    df['HLFPA'] = (7 * df['File'] + 5 * df['Interface'] +
                   4 * df['Input'] + 5 * df['Output'] + 4 * df['Enquiry'])

    # IFPA = Indicative FPA: uses only data BFCs (ILF and EIF)
    # ILF(File)=35, EIF(Interface)=15
    df['IFPA'] = 35 * df['File'] + 15 * df['Interface']

    effort = df['Effort'].values
    mean_effort = float(np.mean(effort))
    median_effort = float(np.median(effort))

    mars = {}
    mdapes = {}
    all_residuals = {}  # store absolute residuals for Wilcoxon tests

    # --- Baselines (Section 3.3 of the paper) ---
    # MeanEFF and MedianEFF evaluated with LOO-CV
    mean_cv_preds = np.zeros_like(effort, dtype=float)
    median_cv_preds = np.zeros_like(effort, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(effort):
        mean_cv_preds[test_idx[0]] = float(np.mean(effort[train_idx]))
        median_cv_preds[test_idx[0]] = float(np.median(effort[train_idx]))

    mars['MeanEFF'] = calculate_mar(effort, mean_cv_preds)
    mdapes['MeanEFF'] = calculate_mdape(effort, mean_cv_preds)
    all_residuals['MeanEFF'] = np.abs(effort - mean_cv_preds)

    mars['MedianEFF'] = calculate_mar(effort, median_cv_preds)
    mdapes['MedianEFF'] = calculate_mdape(effort, median_cv_preds)
    all_residuals['MedianEFF'] = np.abs(effort - median_cv_preds)

    # --- SLR Models (Section 3.2 of the paper) ---
    size_metrics = ['AFP', 'UFP', 'HLFPA', 'IFPA']
    diagnostics = {}

    for size_name in size_metrics:
        X = df[size_name].values

        # LOO-CV predictions
        cv_preds = loo_cv_log_slr_predictions(X, effort)
        model_key = f'slr_{size_name}'

        mars[model_key] = calculate_mar(effort, cv_preds)
        mdapes[model_key] = calculate_mdape(effort, cv_preds)
        all_residuals[model_key] = np.abs(effort - cv_preds)

        # Fit full model (for production predictions)
        models[model_key] = fit_log_slr(X, effort)

        # Model diagnostics (Section 3.2)
        diagnostics[model_key] = model_diagnostics(X, effort)

    # --- Statistical Tests (Section 3.5 of the paper) ---
    # Wilcoxon signed-rank tests with Bonferroni correction
    # For RQ1/RQ2: each model vs. both baselines → k = 2 comparisons per model
    # Bonferroni: α_adj = 0.05 / (number of comparisons per family)
    n_comparisons = 2  # each model is compared against 2 baselines
    alpha = 0.05
    alpha_bonferroni = alpha / n_comparisons

    hypothesis_tests = {}
    for model_key in ['slr_AFP', 'slr_UFP', 'slr_HLFPA', 'slr_IFPA']:
        tests = {}
        for baseline_key in ['MeanEFF', 'MedianEFF']:
            p_val = wilcoxon_test(all_residuals[model_key], all_residuals[baseline_key])
            a12 = vargha_delaney_a12(all_residuals[model_key], all_residuals[baseline_key])
            effect = classify_effect_size(a12)
            significant = p_val < alpha_bonferroni

            tests[baseline_key] = {
                'p_value': round(p_val, 6),
                'a12': round(a12, 4),
                'effect_size': effect,
                'significant': significant,
                'alpha_bonferroni': round(alpha_bonferroni, 4),
            }
        hypothesis_tests[model_key] = tests

    # --- Cross-model comparisons (RQ3 analogue) ---
    # Compare approximate methods (HLFPA, IFPA, UFP) against AFP (the standard FPA measure in this dataset)
    model_comparisons = {}
    for approx_key in ['slr_UFP', 'slr_HLFPA', 'slr_IFPA']:
        p_val = wilcoxon_test(all_residuals[approx_key], all_residuals['slr_AFP'])
        a12 = vargha_delaney_a12(all_residuals[approx_key], all_residuals['slr_AFP'])
        # Two-sided test for RQ3: are they different?
        try:
            _, p_two = scipy_stats.wilcoxon(
                all_residuals['slr_AFP'] - all_residuals[approx_key],
                alternative='two-sided'
            )
        except ValueError:
            p_two = 1.0
        model_comparisons[approx_key] = {
            'vs': 'slr_AFP',
            'p_value_two_sided': round(float(p_two), 6),
            'a12': round(a12, 4),
            'effect_size': classify_effect_size(a12),
        }

    # --- Assemble final statistics ---
    global dataset_stats
    dataset_stats = {
        'total_projects': len(df),
        'mean_effort': mean_effort,
        'median_effort': median_effort,
        'mean_afp': float(df['AFP'].mean()),
        'mean_ufp': float(df['UFP'].mean()),
        'mars': {k: round(float(v), 2) if np.isfinite(v) else 0.0 for k, v in mars.items()},
        'mdapes': {k: round(v, 2) for k, v in mdapes.items()},
        'diagnostics': diagnostics,
        'hypothesis_tests': hypothesis_tests,
        'model_comparisons': model_comparisons,
        'methodology': {
            'paper': 'Di Martino, Ferrucci, Gravino et al. (IST 2020)',
            'dataset': 'China Software Benchmark (499 projects, cross-company)',
            'technique': 'Simple Linear Regression (log-log)',
            'validation': 'Leave-One-Out Cross Validation',
            'primary_metric': 'MAR (Mean of Absolute Residuals)',
            'statistical_test': 'Wilcoxon signed-rank test (one-tailed)',
            'effect_size': 'Vargha-Delaney A12',
            'correction': f'Bonferroni (α = {alpha}/{n_comparisons} = {alpha_bonferroni})',
            'note': 'COSMIC methods not replicable on this dataset (no COSMIC data available).',
        }
    }

    print("[OK] All predictive models successfully trained and loaded in memory.")
    print(f"     Dataset: {len(df)} projects | Metrics: AFP, UFP, HLFPA, IFPA")
    print(f"     Baselines: MeanEFF MAR={mars['MeanEFF']:.0f}, MedianEFF MAR={mars['MedianEFF']:.0f}")
    for sm in size_metrics:
        k = f'slr_{sm}'
        d = diagnostics[k]
        print(f"     {k}: MAR={mars[k]:.0f}, R²={d['r2']}, F={d['f_stat']}, Sign.F={d['sign_f']}")

# Train models on startup
train_models()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html', stats=dataset_stats)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract inputs and convert to float/int
        input_data = {
            'Input': float(data.get('Input', 0)),
            'Output': float(data.get('Output', 0)),
            'Enquiry': float(data.get('Enquiry', 0)),
            'File': float(data.get('File', 0)),
            'Interface': float(data.get('Interface', 0)),
        }

        # Compute size metrics
        ufp = input_data['Input'] + input_data['Output'] + input_data['Enquiry'] + input_data['File'] + input_data['Interface']
        # AFP approximation: UFP × 1.06 (average VAF factor)
        # NOTE: The original China dataset contains AFP measured with project-specific VAF.
        # For new projects without a known VAF, we use the ISBSG average adjustment of ~1.06.
        afp = ufp * 1.06

        # HLFPA: High-Level FPA with average complexity weights
        hlfpa = (7 * input_data['File'] + 5 * input_data['Interface'] +
                 4 * input_data['Input'] + 5 * input_data['Output'] + 4 * input_data['Enquiry'])
        # IFPA: Indicative FPA using only data BFCs
        ifpa = 35 * input_data['File'] + 15 * input_data['Interface']

        predictions = {}
        is_empty_project = (ufp == 0 and afp == 0)

        if is_empty_project:
            predictions['slr_AFP'] = 0.0
            predictions['slr_UFP'] = 0.0
            predictions['slr_HLFPA'] = 0.0
            predictions['slr_IFPA'] = 0.0
        else:
            def predict_slr(model_name, size_val):
                if size_val <= 0:
                    return 0.0
                safe_size = max(size_val, np.finfo(float).eps)
                log_pred = models[model_name].predict(np.log([[safe_size]]))[0]
                return float(np.exp(log_pred))

            predictions['slr_AFP'] = predict_slr('slr_AFP', afp)
            predictions['slr_UFP'] = predict_slr('slr_UFP', ufp)
            predictions['slr_HLFPA'] = predict_slr('slr_HLFPA', hlfpa)
            predictions['slr_IFPA'] = predict_slr('slr_IFPA', ifpa)

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
            'diagnostics': dataset_stats['diagnostics'],
            'hypothesis_tests': dataset_stats['hypothesis_tests'],
            'model_comparisons': dataset_stats['model_comparisons'],
            'methodology': dataset_stats['methodology'],
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    # Serve locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
