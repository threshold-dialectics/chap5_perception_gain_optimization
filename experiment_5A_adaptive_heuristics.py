
#experiment_1_adaptive_heuristics_refactored.py
import numpy as np
import matplotlib.pyplot as plt

# Double all default font sizes to improve figure readability
BASE_FONT_SIZE = plt.rcParams.get("font.size", 10) * 2
for key in [
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "figure.titlesize",
]:
    plt.rcParams[key] = BASE_FONT_SIZE
import pandas as pd
import itertools
import time
import seaborn as sns # For pairplot
import os # For creating results folder
import joblib # For saving trained models
import inspect
import sys
import platform
import scipy.stats
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False
# Scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as model_evaluation_mse
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyRegressor
import sklearn

# --- Results Folder (must be defined before SummaryWriter uses it) ---
RESULTS_FOLDER = "results_experiment_5A_adaptive_heuristic"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Collect important metrics for quick reference at the end
metrics_table_records = []

# --- Summary Writer Class ---
class SummaryWriter:
    def __init__(self, filepath, print_to_console=True):
        self.filepath = filepath
        self.print_to_console = print_to_console
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.file_handle = open(self.filepath, 'w', encoding='utf-8')
        if self.print_to_console:
            print(f"Summary will be saved to: {self.filepath}")

    def write(self, message, end="\n", console_only=False):
        if self.print_to_console:
            print(str(message), end=end)
        if not console_only and self.file_handle:
            self.file_handle.write(str(message) + end)
            self.file_handle.flush() # Ensure it's written immediately

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            if self.print_to_console:
                print(f"Summary file closed: {self.filepath}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# For SHAP value analysis (optional, install if needed: pip install shap)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # This print is for immediate console feedback before summary writer is active in main.
    print("SHAP library not found. Install with 'pip install shap' for SHAP value analysis.")


# --- Configuration Parameters ---
DEFAULT_FREQUENCY_S_TRUE = 1.0
DURATION_T = 2.0
SAMPLING_RATE_T = 100
NUM_POINTS = int(DURATION_T * SAMPLING_RATE_T)
T_VECTOR = np.linspace(0, DURATION_T, NUM_POINTS, endpoint=False)

G_LEVER_VALUES_FOR_SWEEP = np.linspace(0.1, 3.0, 30) # Reduced range and points for gLever for faster full grid
NUM_TRIALS_PER_G_VALUE = 20 # Increased for more accurate optima per combination


# --- Helper Functions (Signal Generation, Noise, Processing Chain, Metrics) ---
def generate_true_signal(t_vector, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t_vector)

def generate_input_noise(num_points, sigma_in_sq):
    std_dev_in = np.sqrt(max(0, sigma_in_sq))
    return np.random.normal(0, std_dev_in, num_points)

def generate_gain_noise(num_points, g_lever, alpha_gain_noise, k_gain, sigma_in_sq_base):
    if g_lever < 1e-9: var_Ng = 0.0
    else: var_Ng = k_gain * (g_lever**alpha_gain_noise) * sigma_in_sq_base
    std_dev_Ng = np.sqrt(max(0, var_Ng))
    return np.random.normal(0, std_dev_Ng, num_points)

def clip_signal(signal, s_max):
    return np.clip(signal, -s_max, s_max)

def run_signal_processing_chain(S_true_segment, g_lever, sigma_in_sq, k_gain, alpha_gain_noise, s_max):
    num_points_segment = len(S_true_segment)
    N_in = generate_input_noise(num_points_segment, sigma_in_sq)
    S_obs = S_true_segment + N_in
    S_gained = g_lever * S_obs
    N_g = generate_gain_noise(num_points_segment, g_lever, alpha_gain_noise, k_gain, sigma_in_sq)
    S_noisy_gained = S_gained + N_g
    S_saturated = clip_signal(S_noisy_gained, s_max)
    return S_saturated

def calculate_mse(S_true, S_hat):
    return np.mean((S_true - S_hat)**2)

def calculate_snr_eff(S_true_var, S_true, S_hat):
    error_signal = S_hat - S_true
    var_error_signal = np.var(error_signal)
    epsilon = 1e-12
    if S_true_var < epsilon: return np.nan
    if var_error_signal < epsilon: return np.inf
    return S_true_var / var_error_signal

# --- Additional Utility Functions for Summary Statistics ---
def approx_hartigans_diptest(data, n_boot=200):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    data = np.sort(data)
    n = len(data)
    if n < 3:
        return 0.0, 1.0
    cdf_emp = np.linspace(1.0 / n, 1.0, n)
    best_dip = np.inf
    for m in range(1, n - 1):
        ir_inc = IsotonicRegression(increasing=True)
        ir_dec = IsotonicRegression(increasing=False)
        fit_left = ir_inc.fit_transform(data[: m + 1], cdf_emp[: m + 1])
        fit_right = ir_dec.fit_transform(data[m:], cdf_emp[m:])
        fit = np.concatenate([fit_left, fit_right])
        diff = np.max(np.abs(fit - cdf_emp))
        if diff < best_dip:
            best_dip = diff
    boot_dips = []
    for _ in range(n_boot):
        boot = np.sort(np.random.uniform(0, 1, n))
        boot_cdf = np.linspace(1.0 / n, 1.0, n)
        best_boot = np.inf
        for m in range(1, n - 1):
            fit_left = IsotonicRegression(increasing=True).fit_transform(
                boot[: m + 1], boot_cdf[: m + 1]
            )
            fit_right = IsotonicRegression(increasing=False).fit_transform(
                boot[m:], boot_cdf[m:]
            )
            diff = np.max(np.abs(np.concatenate([fit_left, fit_right]) - boot_cdf))
            if diff < best_boot:
                best_boot = diff
        boot_dips.append(best_boot)
    p_val = float(np.mean(np.array(boot_dips) >= best_dip))
    return float(best_dip), p_val


def count_kde_peaks(data, num_points=200):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(np.unique(data)) < 2:
        return 1
    kde = scipy.stats.gaussian_kde(data)
    grid = np.linspace(data.min(), data.max(), num_points)
    pdf = kde(grid)
    grad = np.diff(pdf)
    signs = np.sign(grad)
    changes = np.diff(signs)
    peaks = np.sum(changes < 0)
    return int(peaks)


def cooks_distance_95th(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    mse = np.mean(resid ** 2)
    hat = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    p = X.shape[1]
    cd = (resid ** 2) / (p * mse) * (hat / (1 - hat) ** 2)
    cd_95 = np.percentile(cd, 95)
    count_high = int(np.sum(cd > 4 / len(x)))
    return float(cd_95), count_high


def silverman_bandwidth(data):
    data = np.asarray(data)
    if len(data) == 0:
        return np.nan
    sd = np.std(data, ddof=1)
    n = len(data)
    return float(1.06 * sd * n ** (-0.2))

def distance_correlation(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    a = squareform(pdist(x.reshape(-1, 1)))
    b = squareform(pdist(y.reshape(-1, 1)))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    if dvar_x <= 0 or dvar_y <= 0:
        return np.nan
    return float(np.sqrt(dcov2) / np.sqrt(np.sqrt(dvar_x * dvar_y)))

def point_cloud_metrics(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    points = np.column_stack([x, y])
    try:
        hull = ConvexHull(points)
        hull_area = float(hull.volume)
    except Exception:
        hull_area = np.nan
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    chi2_val = scipy.stats.chi2.ppf(0.95, 2)
    widths = 2 * np.sqrt(eigvals * chi2_val)
    orientation = float(np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0])))
    return hull_area, float(widths[0]), float(widths[1]), orientation

def calibration_curve_regression(y_true, y_pred, n_bins=10):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df['bin'] = pd.qcut(df['pred'], q=n_bins, duplicates='drop')
    grouped = df.groupby('bin')
    preds = grouped['pred'].mean().values
    obs = grouped['true'].mean().values
    return preds, obs

# --- Main Experiment Sweep Function (Inner Loop) ---
def run_gLever_sweep_for_params(
    g_lever_values_to_sweep, num_trials_per_g,
    sigma_in_sq, k_gain, alpha_gain_noise, s_max,
    t_vector, amplitude_s_true, frequency_s_true
):
    all_mse_results = []
    S_true = generate_true_signal(t_vector, amplitude_s_true, frequency_s_true)
    for g_lever in g_lever_values_to_sweep:
        trial_mses = []
        for _ in range(num_trials_per_g):
            S_hat = run_signal_processing_chain(
                S_true, g_lever,
                sigma_in_sq, k_gain, alpha_gain_noise, s_max
            )
            trial_mses.append(calculate_mse(S_true, S_hat))
        all_mse_results.append(np.mean(trial_mses))
    return np.array(all_mse_results)

# --- Feature Engineering ---
def add_engineered_features(df):
    df_eng = df.copy()
    epsilon = 1e-9
    df_eng['input_snr_proxy'] = (df_eng['amplitude_s_true']**2) / (df_eng['sigma_in_sq'] + epsilon)
    df_eng['saturation_headroom_ratio'] = df_eng['s_max'] / (df_eng['amplitude_s_true'] + epsilon)
    df_eng['gain_noise_sensitivity_proxy'] = df_eng['k_gain'] * df_eng['alpha_gain_noise'] # Simple interaction
    return df_eng

# --- Model Training, Evaluation, and Hyperparameter Tuning Function ---
def train_evaluate_tune_model(model_constructor, X, y, model_name, summary,
                              param_dist=None, cv_folds=10, n_iter_search=20,
                              random_state=42, feature_names_list=None):
    summary.write(f"\n--- Training, Evaluating, and Tuning: {model_name} ---")
    if X.empty or y.empty:
        summary.write("  Cannot train model: X or y is empty.")
        return None, np.nan, {}

    y_ravel = y.ravel()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    best_model = model_constructor()
    best_cv_mse = np.inf
    tuned_params = {}
    cv_metrics = None

    if param_dist: # Perform hyperparameter tuning if param_dist is provided
        summary.write("  Performing RandomizedSearchCV for hyperparameter tuning...")
        random_search = RandomizedSearchCV(model_constructor(), param_distributions=param_dist,
                                           n_iter=n_iter_search, cv=kf,
                                           scoring='neg_mean_squared_error', random_state=random_state, n_jobs=-1)
        try:
            random_search.fit(X, y_ravel)
            best_model = random_search.best_estimator_
            best_cv_mse = -random_search.best_score_  # best_score_ is neg_mse
            tuned_params = random_search.best_params_
            summary.write(f"  Best Tuned Parameters: {tuned_params}")
            summary.write(f"  Best Cross-Validated MSE (from tuning): {best_cv_mse:.4f}")
            try:
                cv_neg_mse_scores = cross_val_score(best_model, X, y_ravel, cv=kf,
                                                   scoring='neg_mean_squared_error')
                cv_mae_scores = -cross_val_score(best_model, X, y_ravel, cv=kf,
                                                scoring='neg_mean_absolute_error')
                cv_r2_scores = cross_val_score(best_model, X, y_ravel, cv=kf, scoring='r2')
                cv_metrics = {
                    'mse': (-cv_neg_mse_scores).tolist(),
                    'mae': cv_mae_scores.tolist(),
                    'r2': cv_r2_scores.tolist()
                }
                summary.write(
                    f"  CV MSE mean±SD: {np.mean(-cv_neg_mse_scores):.4f} ± {np.std(-cv_neg_mse_scores):.4f}; "
                    f"MAE mean±SD: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}; "
                    f"R² mean±SD: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}"
                )
            except Exception as e_cv:
                summary.write(f"  Error computing CV metric distribution: {e_cv}")
        except Exception as e:
            summary.write(f"  Error during RandomizedSearchCV for {model_name}: {e}")
            summary.write("  Falling back to default model parameters.")
            best_model = model_constructor() # Re-initialize to default
            try:
                cv_neg_mse_scores = cross_val_score(best_model, X, y_ravel, cv=kf, scoring='neg_mean_squared_error')
                best_cv_mse = -np.mean(cv_neg_mse_scores)
                summary.write(f"  Cross-Validated MSE (default params): {best_cv_mse:.4f}")
            except Exception as e_cv:
                summary.write(f"  Error during CV with default params for {model_name}: {e_cv}")
                best_cv_mse = np.nan
    else: # No hyperparameter tuning, just CV with default parameters
        summary.write("  Evaluating with default parameters (no tuning)...")
        try:
            cv_neg_mse_scores = cross_val_score(best_model, X, y_ravel, cv=kf, scoring='neg_mean_squared_error')
            cv_mae_scores = -cross_val_score(best_model, X, y_ravel, cv=kf, scoring='neg_mean_absolute_error')
            cv_r2_scores = cross_val_score(best_model, X, y_ravel, cv=kf, scoring='r2')
            best_cv_mse = -np.mean(cv_neg_mse_scores)
            cv_metrics = {
                'mse': (-cv_neg_mse_scores).tolist(),
                'mae': cv_mae_scores.tolist(),
                'r2': cv_r2_scores.tolist()
            }
            summary.write(
                f"  CV MSE mean±SD: {np.mean(-cv_neg_mse_scores):.4f} ± {np.std(-cv_neg_mse_scores):.4f}; "
                f"MAE mean±SD: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}; "
                f"R² mean±SD: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}"
            )
        except Exception as e:
            summary.write(f"  Error during cross-validation for {model_name}: {e}")
            best_cv_mse = np.nan

    # Fit the best (or default) model on the entire training data
    final_model = None
    if best_model: # best_model is an instance here
        try:
            final_model = best_model.fit(X, y_ravel)
            estimator_to_inspect = final_model.steps[-1][1] if isinstance(final_model, Pipeline) else final_model
            f_names = feature_names_list if feature_names_list is not None else (X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])])

            if hasattr(estimator_to_inspect, 'coef_'): summary.write(f"  Coefficients: {estimator_to_inspect.coef_}")
            if hasattr(estimator_to_inspect, 'intercept_'): summary.write(f"  Intercept: {estimator_to_inspect.intercept_:.4f}")
            if hasattr(estimator_to_inspect, 'feature_importances_'):
                importances = pd.Series(estimator_to_inspect.feature_importances_, index=f_names)
                summary.write("  Feature Importances (from model):\n" + importances.sort_values(ascending=False).to_string())
        except Exception as e:
            summary.write(f"  Error during final model fitting for {model_name}: {e}")
            final_model = None

    return final_model, best_cv_mse, tuned_params, cv_metrics

# --- Visualization Function for Contour Plots ---
def plot_heuristic_behavior(model,
                            df_full,
                            feature_names_orig,
                            engineered_feature_names,
                            results_folder,
                            summary):
    summary.write("\n--- Characterizing Heuristic Model Behavior (Contour Plots) ---")
    if model is None or df_full.empty:
        summary.write("  Model or data not available for plotting.")
        return

    feature_cols = feature_names_orig + engineered_feature_names
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in df_full.columns]

    if {'input_snr_proxy', 'saturation_headroom_ratio'}.issubset(df_full.columns):
        feat1_name, feat2_name = 'input_snr_proxy', 'saturation_headroom_ratio'
    else:
        if len(feature_cols) < 2:
            summary.write("  Not enough features for 2-D contour plot.")
            return
        feat1_name, feat2_name = feature_cols[:2] # Fallback to first two available features
    summary.write(f"  Generating contour plot for features: '{feat1_name}' vs '{feat2_name}' with other features at mean.")

    f1_vals = np.linspace(df_full[feat1_name].min(), df_full[feat1_name].max(), 30)
    f2_vals = np.linspace(df_full[feat2_name].min(), df_full[feat2_name].max(), 30)
    f1_grid, f2_grid = np.meshgrid(f1_vals, f2_vals)

    mean_vector = df_full[feature_cols].mean()
    grid_records = []
    for f1_val, f2_val in itertools.product(f1_vals, f2_vals): # Iterate properly over grid points
        row = mean_vector.copy()
        row[feat1_name] = f1_val
        row[feat2_name] = f2_val
        grid_records.append(row)
    
    grid_df = pd.DataFrame(grid_records, columns=feature_cols) # Ensure column order
    
    # Reshape Z_pred correctly based on the order of f1_vals and f2_vals in itertools.product
    # itertools.product(f1_vals, f2_vals) means f1_vals changes fastest.
    # This corresponds to (row, col) indexing if f1 is y-axis and f2 is x-axis, or vice-versa.
    # The meshgrid f1_grid, f2_grid are (len(f2_vals), len(f1_vals)).
    # Z_pred needs to be reshaped to (len(f2_vals), len(f1_vals)) if f1_vals are columns and f2_vals are rows.
    #plt.contourf(X, Y, Z) expects Z to have shape (Y.shape[0], X.shape[0])
    # f1_grid is (len(f2_vals), len(f1_vals)), f2_grid is (len(f2_vals), len(f1_vals))
    # Z_pred should be (len(f2_vals), len(f1_vals))

    try:
        Z_pred_flat = model.predict(grid_df[feature_cols])
        Z_pred = Z_pred_flat.reshape(len(f2_vals), len(f1_vals)) # f2_vals corresponds to rows, f1_vals to columns
    except Exception as e:
        summary.write(f"  Error generating predictions for contour: {e}")
        return

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(f1_grid, f2_grid, Z_pred, levels=20, cmap='viridis')
    plt.colorbar(cp, label='Predicted optimal g_mse')
    plt.scatter(df_full[feat1_name], df_full[feat2_name],
                c=df_full['optimal_g_mse'], cmap='viridis_r',
                s=20, edgecolor='k', alpha=0.5, label='Simulated optimum')
    plt.xlabel(feat1_name)
    plt.ylabel(feat2_name)
    plt.title('Heuristic model: predicted optimum g')
    plt.legend()
    out_path = os.path.join(results_folder, "heuristic_contour_plot.png")
    plt.savefig(out_path)
    plt.close()
    summary.write(f"  Contour plot saved to {out_path}")
    try:
        summary.write(
            f"FIG_contour_{feat1_name}_vs_{feat2_name}: file={out_path}, x={feat1_name} ({f1_vals.min():.3g}-{f1_vals.max():.3g}), "
            f"y={feat2_name} ({f2_vals.min():.3g}-{f2_vals.max():.3g}), n={len(df_full)}"
        )
    except Exception:
        pass
    try:
        summary.write(
            f"FIG_contour_axes: x=[{f1_vals.min():.3g},{f1_vals.max():.3g}], "
            f"y=[{f2_vals.min():.3g},{f2_vals.max():.3g}]; n_points={len(df_full)}"
        )
    except Exception:
        pass

    # --- Textual Summary of Contour Plot ---
    summary.write(f"  Textual Summary of Contour Plot ('{feat1_name}' vs. '{feat2_name}'):")
    summary.write(f"    The plot visualizes the heuristic model's predicted 'optimal_g_mse' values.")
    summary.write(f"    '{feat1_name}' (x-axis) ranges from {f1_vals.min():.3g} to {f1_vals.max():.3g}.")
    summary.write(f"    '{feat2_name}' (y-axis) ranges from {f2_vals.min():.3g} to {f2_vals.max():.3g}.")
    
    min_Z_pred_val = np.nanmin(Z_pred)
    max_Z_pred_val = np.nanmax(Z_pred)
    summary.write(f"    Predicted 'optimal_g_mse' on this grid ranges from {min_Z_pred_val:.3f} to {max_Z_pred_val:.3f}.")

    min_idx_unraveled = np.unravel_index(np.nanargmin(Z_pred), Z_pred.shape)
    max_idx_unraveled = np.unravel_index(np.nanargmax(Z_pred), Z_pred.shape)
    
    # f1_grid and f2_grid have shape (len(f2_vals), len(f1_vals))
    # So min_idx_unraveled[0] corresponds to f2_vals, min_idx_unraveled[1] to f1_vals
    f1_at_min_Z = f1_grid[min_idx_unraveled]
    f2_at_min_Z = f2_grid[min_idx_unraveled]
    f1_at_max_Z = f1_grid[max_idx_unraveled]
    f2_at_max_Z = f2_grid[max_idx_unraveled]

    summary.write(f"    The minimum predicted 'optimal_g_mse' ({min_Z_pred_val:.3f}) occurs at approximately {feat1_name}={f1_at_min_Z:.3g} and {feat2_name}={f2_at_min_Z:.3g}.")
    summary.write(f"    The maximum predicted 'optimal_g_mse' ({max_Z_pred_val:.3f}) occurs at approximately {feat1_name}={f1_at_max_Z:.3g} and {feat2_name}={f2_at_max_Z:.3g}.")

    # Gradient magnitudes at a few interior points
    grad_points = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    for gx, gy in grad_points:
        xi = int(gx * (len(f1_vals) - 1))
        yi = int(gy * (len(f2_vals) - 1))
        gradx = np.nan
        grady = np.nan
        try:
            if 0 < xi < len(f1_vals) - 1:
                gradx = (Z_pred[yi, xi + 1] - Z_pred[yi, xi - 1]) / (f1_vals[xi + 1] - f1_vals[xi - 1])
            if 0 < yi < len(f2_vals) - 1:
                grady = (Z_pred[yi + 1, xi] - Z_pred[yi - 1, xi]) / (f2_vals[yi + 1] - f2_vals[yi - 1])
        except Exception:
            gradx, grady = np.nan, np.nan
        mag = np.sqrt(gradx**2 + grady**2) if not np.isnan(gradx) and not np.isnan(grady) else np.nan
        summary.write(
            f"    Gradient magnitude at ~({gx:.2f},{gy:.2f}) of grid: {mag:.4f}"
        )

    # Quadratic surface fit around minimum
    try:
        idx_min_x = int(np.clip(min_idx_unraveled[1], 1, len(f1_vals)-2))
        idx_min_y = int(np.clip(min_idx_unraveled[0], 1, len(f2_vals)-2))
        window_x = slice(idx_min_x-1, idx_min_x+2)
        window_y = slice(idx_min_y-1, idx_min_y+2)
        X_quad = []
        y_quad = []
        for i_y in range(window_y.start, window_y.stop):
            for i_x in range(window_x.start, window_x.stop):
                X_quad.append([1, f1_vals[i_x], f2_vals[i_y], f1_vals[i_x]**2, f1_vals[i_x]*f2_vals[i_y], f2_vals[i_y]**2])
                y_quad.append(Z_pred[i_y, i_x])
        X_quad = np.array(X_quad)
        y_quad = np.array(y_quad)
        coeffs, *_ = np.linalg.lstsq(X_quad, y_quad, rcond=None)
        summary.write("    Local quadratic fit around minimum: Z = a + b1*x + b2*y + c1*x^2 + c2*x*y + c3*y^2")
        summary.write(
            f"      Coefficients: a={coeffs[0]:.4f}, b1={coeffs[1]:.4f}, b2={coeffs[2]:.4f}, "
            f"c1={coeffs[3]:.4f}, c2={coeffs[4]:.4f}, c3={coeffs[5]:.4f}"
        )
    except Exception as e_quad:
        summary.write(f"    Could not fit quadratic surface near minimum: {e_quad}")

    # Gradient information near the center of the contour grid
    grad_f1 = np.nan
    grad_f2 = np.nan
    if len(f1_vals) > 2 and len(f2_vals) > 2:
        c_idx = len(f1_vals) // 2
        r_idx = len(f2_vals) // 2
        try:
            if 0 < c_idx < len(f1_vals) - 1:
                grad_f1 = (Z_pred[r_idx, c_idx + 1] - Z_pred[r_idx, c_idx - 1]) / (f1_vals[c_idx + 1] - f1_vals[c_idx - 1])
            if 0 < r_idx < len(f2_vals) - 1:
                grad_f2 = (Z_pred[r_idx + 1, c_idx] - Z_pred[r_idx - 1, c_idx]) / (f2_vals[r_idx + 1] - f2_vals[r_idx - 1])
        except Exception:
            grad_f1, grad_f2 = np.nan, np.nan
    summary.write(
        f"    Gradient near grid center: d(optimal_g_mse)/d{feat1_name} ≈ {grad_f1:.4f}, d(optimal_g_mse)/d{feat2_name} ≈ {grad_f2:.4f}."
    )

    # Prediction error statistics on the full dataset
    try:
        preds_full = model.predict(df_full[feature_cols])
        abs_errors = np.abs(preds_full - df_full['optimal_g_mse'])
        summary.write(
            f"    Prediction absolute error: mean={abs_errors.mean():.4f}, median={np.median(abs_errors):.4f}, 95th percentile={np.percentile(abs_errors, 95):.4f}."
        )
        r2 = 1 - np.sum(abs_errors**2) / np.sum((df_full['optimal_g_mse'] - df_full['optimal_g_mse'].mean())**2)
        summary.write(
            f"    Fit quality on scatter points: R²={r2:.4f}, RMSE={np.sqrt(np.mean(abs_errors**2)):.4f}"
        )
    except Exception as e:
        summary.write(f"    Could not compute prediction error statistics: {e}")

    # Additional quantitative descriptors of contour surface
    try:
        local_minima = 0
        local_minima_coords = []
        for r in range(1, Z_pred.shape[0]-1):
            for c in range(1, Z_pred.shape[1]-1):
                neighborhood = Z_pred[r-1:r+2, c-1:c+2]
                val = Z_pred[r, c]
                if np.isfinite(val) and val == np.nanmin(neighborhood) and np.sum(neighborhood==val)==1:
                    local_minima += 1
                    local_minima_coords.append((float(f1_grid[r,c]), float(f2_grid[r,c]), float(val)))
        summary.write(f"    Grid search found {local_minima} local minimum{'s' if local_minima!=1 else ''}.")
        if local_minima_coords:
            for x,y,g_val in local_minima_coords:
                summary.write(f"      at ({x:.3g}, {y:.3g}) -> g={g_val:.4f}")
        grad_y, grad_x = np.gradient(Z_pred, f2_vals, f1_vals)
        hxx = np.gradient(grad_x, f1_vals, axis=1)
        hyy = np.gradient(grad_y, f2_vals, axis=0)
        hxy = np.gradient(grad_x, f2_vals, axis=0)
        hyx = np.gradient(grad_y, f1_vals, axis=1)
        hessian = np.array([[hxx[min_idx_unraveled], hxy[min_idx_unraveled]],
                           [hyx[min_idx_unraveled], hyy[min_idx_unraveled]]], dtype=float)
        eigvals = np.linalg.eigvalsh(hessian)
        summary.write(f"    Local curvature eigenvalues at global minimum: λ1={eigvals[0]:.4f}, λ2={eigvals[1]:.4f}")
        mid_x_low = int(Z_pred.shape[1]*0.25)
        mid_x_high = int(Z_pred.shape[1]*0.75)
        mid_y_low = int(Z_pred.shape[0]*0.25)
        mid_y_high = int(Z_pred.shape[0]*0.75)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        central_mag = grad_mag[mid_y_low:mid_y_high, mid_x_low:mid_x_high]
        summary.write(
            f"    ∇g magnitude over full grid: min={np.nanmin(grad_mag):.4f}, median={np.nanmedian(grad_mag):.4f}, max={np.nanmax(grad_mag):.4f}"
        )
        summary.write(
            f"    ∇g magnitude in central 50%: min={np.nanmin(central_mag):.4f}, median={np.nanmedian(central_mag):.4f}, max={np.nanmax(central_mag):.4f}"
        )
        within_5pct_mask = Z_pred <= 1.05 * min_Z_pred_val
        n_within_5pct = int(np.sum(within_5pct_mask))
        area95 = n_within_5pct / Z_pred.size
        valley_width_x = np.ptp(f1_grid[within_5pct_mask])
        valley_width_y = np.ptp(f2_grid[within_5pct_mask])
        summary.write(
            f"    Region where predicted optimum ≤ 1.05×g* covers {area95*100:.1f}% of the domain ({n_within_5pct} grid points)."
        )

        half_level = min_Z_pred_val + 0.5 * (max_Z_pred_val - min_Z_pred_val)
        mask_half = Z_pred <= half_level
        if np.any(mask_half):
            fwhm_x = float(f1_grid[mask_half].max() - f1_grid[mask_half].min())
            fwhm_y = float(f2_grid[mask_half].max() - f2_grid[mask_half].min())
        else:
            fwhm_x = np.nan
            fwhm_y = np.nan

        points = np.column_stack([f1_grid.ravel(), f2_grid.ravel()])
        weights = np.exp(-(Z_pred.ravel() - min_Z_pred_val))
        cov = np.cov(points.T, aweights=weights)
        eigvals_c, eigvecs_c = np.linalg.eigh(cov)
        idx_max = np.argmax(eigvals_c)
        theta_deg = float(np.degrees(np.arctan2(eigvecs_c[1, idx_max], eigvecs_c[0, idx_max])))
        eig_sorted = np.sort(np.abs(eigvals_c))[::-1]
        eig1, eig2 = eig_sorted[:2]
        eccentricity = np.sqrt(1 - eig2 / eig1) if eig1 != 0 else np.nan
        if n_within_5pct > 0:
            centroid_x = float(np.average(f1_grid, weights=within_5pct_mask))
            centroid_y = float(np.average(f2_grid, weights=within_5pct_mask))
        else:
            centroid_x = np.nan
            centroid_y = np.nan
        summary.write(
            f"    FWHM_x≈{fwhm_x:.3g}, FWHM_y≈{fwhm_y:.3g}; valley θ≈{theta_deg:.1f}°, area95≈{area95*100:.1f}%"
        )
        summary.write(
            f"    Valley width x={valley_width_x:.3g}, y={valley_width_y:.3g}, orientation θ={theta_deg:.1f}°"
        )
        summary.write(
            f"    Valley eccentricity e={eccentricity:.3f}; centroid≈({centroid_x:.3g}, {centroid_y:.3g})"
        )
        metrics_table_records.append(("Contour", "FWHMx", fwhm_x))
        metrics_table_records.append(("Contour", "FWHMy", fwhm_y))
        metrics_table_records.append(("Contour", "Valley_theta_deg", theta_deg))
        metrics_table_records.append(("Contour", "Area95_pct", area95*100))
        metrics_table_records.append(("Contour", "Valley_width_x", valley_width_x))
        metrics_table_records.append(("Contour", "Valley_width_y", valley_width_y))
        metrics_table_records.append(("Contour", "N_pts_within_5pct", n_within_5pct))
        metrics_table_records.append(("Contour", "Eccentricity", eccentricity))
        metrics_table_records.append(("Contour", "Centroid_x", centroid_x))
        metrics_table_records.append(("Contour", "Centroid_y", centroid_y))

        try:
            points_xy = df_full[[feat1_name, feat2_name]].dropna().values
            if len(points_xy) >= 3:
                hull = ConvexHull(points_xy)
                hull_area = float(hull.volume)
                centroid = points_xy[hull.vertices].mean(axis=0)
                dists = np.linalg.norm(points_xy - centroid, axis=1)
                mean_dist = float(np.mean(dists))
                pct95_dist = float(np.percentile(dists, 95))
            else:
                hull_area = np.nan
                mean_dist = np.nan
                pct95_dist = np.nan
            summary.write(f"    Convex hull area of simulated optima: {hull_area:.4f}")
            summary.write(
                f"    Mean distance from hull centroid: {mean_dist:.4f}, 95th percentile: {pct95_dist:.4f}"
            )
            metrics_table_records.append(("Contour", "Hull_area", hull_area))
            metrics_table_records.append(("Contour", "Hull_mean_dist", mean_dist))
            metrics_table_records.append(("Contour", "Hull_dist_95th", pct95_dist))
        except Exception as e_hull:
            summary.write(f"    Could not compute convex hull area: {e_hull}")

        try:
            preds_grid = model.predict(df_full[feature_cols])
            err_sq = (preds_grid - df_full['optimal_g_mse']) ** 2
            bins_x = np.linspace(df_full[feat1_name].min(), df_full[feat1_name].max(), 11)
            bins_y = np.linspace(df_full[feat2_name].min(), df_full[feat2_name].max(), 11)
            sum_err, _, _ = np.histogram2d(df_full[feat1_name], df_full[feat2_name], bins=[bins_x, bins_y], weights=err_sq)
            count_err, _, _ = np.histogram2d(df_full[feat1_name], df_full[feat2_name], bins=[bins_x, bins_y])
            mse_grid = np.divide(sum_err, count_err, out=np.full_like(sum_err, np.nan), where=count_err>0)
            rmse_grid = np.sqrt(mse_grid)
            mean_rmse = float(np.nanmean(rmse_grid))
            pct95_rmse = float(np.nanpercentile(rmse_grid, 95))
            summary.write(f"    10x10 error grid RMSE: mean={mean_rmse:.4f}, 95th pct={pct95_rmse:.4f}")
            metrics_table_records.append(("Contour", "RMSEgrid_mean", mean_rmse))
            metrics_table_records.append(("Contour", "RMSEgrid_95th", pct95_rmse))
        except Exception as e_err:
            summary.write(f"    Could not compute error heatmap: {e_err}")
    except Exception as e_desc:
        summary.write(f"    Could not compute advanced contour statistics: {e_desc}")

    # General trends (simplified by looking at changes along axes at midpoint of other axis)
    # Trend along f1_name (x-axis) at mid f2_name (y-axis)
    mid_f2_idx = Z_pred.shape[0] // 2
    z_slice_f1 = Z_pred[mid_f2_idx, :]
    if not np.all(np.isnan(z_slice_f1)) and len(z_slice_f1) > 1:
        if z_slice_f1[~np.isnan(z_slice_f1)][0] < z_slice_f1[~np.isnan(z_slice_f1)][-1]:
            summary.write(f"    Holding '{feat2_name}' near its mid-range ({f2_grid[mid_f2_idx, 0]:.2g}), predicted 'optimal_g_mse' tends to increase as '{feat1_name}' increases.")
        elif z_slice_f1[~np.isnan(z_slice_f1)][0] > z_slice_f1[~np.isnan(z_slice_f1)][-1]:
            summary.write(f"    Holding '{feat2_name}' near its mid-range ({f2_grid[mid_f2_idx, 0]:.2g}), predicted 'optimal_g_mse' tends to decrease as '{feat1_name}' increases.")
        else:
            summary.write(f"    Holding '{feat2_name}' near its mid-range ({f2_grid[mid_f2_idx, 0]:.2g}), predicted 'optimal_g_mse' shows a complex or flat trend as '{feat1_name}' increases.")

    # Trend along f2_name (y-axis) at mid f1_name (x-axis)
    mid_f1_idx = Z_pred.shape[1] // 2
    z_slice_f2 = Z_pred[:, mid_f1_idx]
    if not np.all(np.isnan(z_slice_f2)) and len(z_slice_f2) > 1:
        if z_slice_f2[~np.isnan(z_slice_f2)][0] < z_slice_f2[~np.isnan(z_slice_f2)][-1]:
            summary.write(f"    Holding '{feat1_name}' near its mid-range ({f1_grid[0, mid_f1_idx]:.2g}), predicted 'optimal_g_mse' tends to increase as '{feat2_name}' increases.")
        elif z_slice_f2[~np.isnan(z_slice_f2)][0] > z_slice_f2[~np.isnan(z_slice_f2)][-1]:
            summary.write(f"    Holding '{feat1_name}' near its mid-range ({f1_grid[0, mid_f1_idx]:.2g}), predicted 'optimal_g_mse' tends to decrease as '{feat2_name}' increases.")
        else:
            summary.write(f"    Holding '{feat1_name}' near its mid-range ({f1_grid[0, mid_f1_idx]:.2g}), predicted 'optimal_g_mse' shows a complex or flat trend as '{feat2_name}' increases.")

    summary.write(f"    Simulated 'optimal_g_mse' values from the original data sweep are overlaid as scatter points.")
    summary.write(f"    A visual inspection of the plot can help assess how well these simulated optima align with the model's predictions across the chosen feature plane.")

# --- Parameter Grid for Systematic Sweep ---
param_grid = {
     'sigma_in_sq': [0.01**2, 0.05**2, 0.1**2, 0.2**2],
     'k_gain': [0.001, 0.01, 0.05, 0.1],
     'alpha_gain_noise': [1.0, 1.5, 2.0],
     's_max': [0.75, 1.0, 2.5, 5.0],
     'amplitude_s_true': [0.5, 1.0, 1.5, 2.0]
}

# --- Main Experiment Execution Function ---
def run_experiment(summary):
    np.random.seed(42)
    start_time_total = time.time()

    summary.write("--- Experiment Run Start ---")
    timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")
    summary.write(f"Timestamp: {timestamp_start}")

    summary.write("\n--- SHAP Library Status ---")
    if SHAP_AVAILABLE:
        summary.write("SHAP library found and available for analysis.")
    else:
        summary.write("SHAP library not found. SHAP value analysis will be skipped.")

    summary.write("\n--- Experiment Configuration ---")
    summary.write(f"Results Folder: {RESULTS_FOLDER}")
    summary.write(f"Default True Signal Frequency: {DEFAULT_FREQUENCY_S_TRUE} Hz")
    summary.write(f"Duration: {DURATION_T} s, Sampling Rate: {SAMPLING_RATE_T} Hz, Num Points: {NUM_POINTS}")
    summary.write(f"gLever Sweep: {len(G_LEVER_VALUES_FOR_SWEEP)} points from {G_LEVER_VALUES_FOR_SWEEP.min():.2f} to {G_LEVER_VALUES_FOR_SWEEP.max():.2f}")
    summary.write(f"Trials per gLever value: {NUM_TRIALS_PER_G_VALUE}")
    summary.write(f"Random seed: 42")

    summary.write("Python and library versions:")
    summary.write(f"  Python: {sys.version.split()[0]}")
    summary.write(f"  numpy: {np.__version__}")
    summary.write(f"  pandas: {pd.__version__}")
    summary.write(f"  scikit-learn: {sklearn.__version__}")
    if SHAP_AVAILABLE:
        summary.write(f"  shap: {shap.__version__}")

    try:
        summary.write(f"CPU: {platform.processor()}")
    except Exception:
        pass
    summary.write("\nParameter Grid for System Sweep:")
    for key, value_list in param_grid.items():
        summary.write(f"  {key}: {value_list}")
    summary.write("-" * 30)

    all_optimal_results_list = []
    param_keys, param_value_levels = zip(*param_grid.items())
    parameter_combinations = [dict(zip(param_keys, v)) for v in itertools.product(*param_value_levels)]

    summary.write(f"\n--- 1. Data Generation (Systematic Parameter Sweep) ---")
    summary.write(f"Starting sweep with NUM_TRIALS_PER_G_VALUE = {NUM_TRIALS_PER_G_VALUE}.")
    summary.write(f"Total parameter combinations to test: {len(parameter_combinations)}")
    estimated_total_sims = len(parameter_combinations) * len(G_LEVER_VALUES_FOR_SWEEP) * NUM_TRIALS_PER_G_VALUE
    summary.write(f"Estimated total core simulations: {estimated_total_sims:,}\n")
    
    for i, current_system_params in enumerate(parameter_combinations):
        progress_msg = f"Running combination {i+1}/{len(parameter_combinations)}..."
        summary.write(progress_msg, end='\r' if (i+1) % (max(1, len(parameter_combinations) // 10)) != 0 and i != 0 else '\n', console_only=True)

        mse_curve = run_gLever_sweep_for_params(
            G_LEVER_VALUES_FOR_SWEEP,
            NUM_TRIALS_PER_G_VALUE,
            **current_system_params,
            t_vector=T_VECTOR,
            frequency_s_true=DEFAULT_FREQUENCY_S_TRUE
        )
        optimal_g_for_mse, min_mse_value = np.nan, np.nan
        if mse_curve.size > 0 and np.any(~np.isnan(mse_curve)):
            valid_mse_idx = ~np.isnan(mse_curve)
            if np.any(valid_mse_idx): 
                min_mse_idx_among_valid = np.argmin(mse_curve[valid_mse_idx])
                optimal_g_for_mse = G_LEVER_VALUES_FOR_SWEEP[valid_mse_idx][min_mse_idx_among_valid]
                min_mse_value = mse_curve[valid_mse_idx][min_mse_idx_among_valid]
        
        result_entry = {**current_system_params, 'optimal_g_mse': optimal_g_for_mse, 'min_mse_at_opt_g': min_mse_value}
        all_optimal_results_list.append(result_entry)
    if len(parameter_combinations)>0 : summary.write(f"Finished all {len(parameter_combinations)} combinations.", console_only=True) 
    summary.write(f"Finished all {len(parameter_combinations)} combinations.")


    results_df = pd.DataFrame(all_optimal_results_list)
    results_df_path = os.path.join(RESULTS_FOLDER, "optimal_gLever_sweep_results.csv")
    results_df.to_csv(results_df_path, index=False)
    summary.write(f"\nFull results of parameter sweep saved to {results_df_path}")
    summary.write(f"Shape of full results DataFrame: {results_df.shape}")
    summary.write(f"Number of NaNs in 'optimal_g_mse': {results_df['optimal_g_mse'].isna().sum()}")


    analysis_df_raw = results_df.dropna(subset=['optimal_g_mse']).copy()
    summary.write(f"Shape of DataFrame after dropping NaNs in 'optimal_g_mse': {analysis_df_raw.shape}")
    analysis_df = add_engineered_features(analysis_df_raw)
    summary.write("Engineered features added: 'input_snr_proxy', 'saturation_headroom_ratio', 'gain_noise_sensitivity_proxy'")

    summary.write("\nDescriptive statistics for all variables:")
    for col in analysis_df.columns:
        series = analysis_df[col]
        summary.write(
            f"  {col}: mean={series.mean():.4g}, sd={series.std():.4g}, median={series.median():.4g}, Q1={series.quantile(0.25):.4g}, Q3={series.quantile(0.75):.4g}"
        )

    summary.write("Counts per level for grid factors:")
    for factor in param_grid.keys():
        counts = analysis_df[factor].value_counts().sort_index()
        summary.write(f"  {factor}: " + ", ".join([f"{idx} -> {cnt}" for idx, cnt in counts.items()]))

    # Verify full factorial coverage
    expected_cells = 1
    for val_list in param_grid.values():
        expected_cells *= len(val_list)
    observed_cells = analysis_df.groupby(list(param_grid.keys())).size().shape[0]
    empty_cells = expected_cells - observed_cells
    summary.write(f"  Fully crossed design check: {observed_cells}/{expected_cells} cells, {empty_cells} empty cells")

    summary.write("\nMean ± SD of optimal_g_mse grouped by single factors:")
    for factor in param_grid.keys():
        grp = analysis_df.groupby(factor)['optimal_g_mse']
        stats = grp.agg(['mean','std'])
        for level, row in stats.iterrows():
            summary.write(f"  {factor}={level}: {row['mean']:.4f} ± {row['std']:.4f}")

    summary.write("\nWithin-level dispersion and Tukey outliers:")
    for factor in param_grid.keys():
        for level, subset in analysis_df.groupby(factor):
            vals = subset['optimal_g_mse'].dropna()
            sd_within = float(vals.std()) if len(vals) > 1 else 0.0
            iqr_within = float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) > 0 else 0.0
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            fence_low = q1 - 1.5 * iqr_within
            fence_high = q3 + 1.5 * iqr_within
            out_n = int(((vals < fence_low) | (vals > fence_high)).sum())
            summary.write(
                f"  {factor}={level}: SD_within={sd_within:.4f}, IQR={iqr_within:.4f}, n_outliers_Tukey={out_n}"
            )
            metrics_table_records.append((f"Dispersion_{factor}", f"{level}_SD", sd_within))
            metrics_table_records.append((f"Dispersion_{factor}", f"{level}_IQR", iqr_within))
            metrics_table_records.append((f"Dispersion_{factor}", f"{level}_Out_n", out_n))

    summary.write("\nQuantiles of optimal_g_mse by factor:")
    for factor in param_grid.keys():
        qtab = analysis_df.groupby(factor)['optimal_g_mse'].quantile([0.1,0.25,0.5,0.75,0.9]).unstack()
        for level, row in qtab.iterrows():
            summary.write(
                f"  {factor}={level}: q10={row[0.1]:.3f}, q25={row[0.25]:.3f}, median={row[0.5]:.3f}, q75={row[0.75]:.3f}, q90={row[0.9]:.3f}"
            )
            metrics_table_records.append((f"Quantile_{factor}", f"{level}_q10", row[0.1]))
            metrics_table_records.append((f"Quantile_{factor}", f"{level}_q25", row[0.25]))
            metrics_table_records.append((f"Quantile_{factor}", f"{level}_q50", row[0.5]))
            metrics_table_records.append((f"Quantile_{factor}", f"{level}_q75", row[0.75]))
            metrics_table_records.append((f"Quantile_{factor}", f"{level}_q90", row[0.9]))

    if 'amplitude_s_true' in analysis_df.columns:
        summary.write("\nInteraction of factors with amplitude_s_true:")
        for factor in param_grid.keys():
            if factor == 'amplitude_s_true':
                continue
            grp = analysis_df.groupby([factor,'amplitude_s_true'])['optimal_g_mse'].agg(['mean','std'])
            for (level, amp), row in grp.iterrows():
                summary.write(f"  {factor}={level}, amp={amp}: {row['mean']:.4f} ± {row['std']:.4f}")
        try:
            means_table = analysis_df.pivot_table(values='optimal_g_mse', index='sigma_in_sq', columns='amplitude_s_true', aggfunc='mean')
            summary.write("\n  Cell means table for sigma_in_sq × amplitude_s_true:")
            for line in means_table.to_string().split('\n'):
                summary.write("  " + line)
        except Exception as e_cell:
            summary.write(f"  Could not compute cell means table: {e_cell}")

    if STATSMODELS_AVAILABLE:
        summary.write("\n--- Factorial ANOVA (main effects and first-order interactions) ---")
        try:
            main_terms = ' + '.join(param_grid.keys())
            inter_terms = ' + '.join([f'{a}:{b}' for i,a in enumerate(param_grid.keys()) for b in list(param_grid.keys())[i+1:]])
            formula = f'optimal_g_mse ~ {main_terms} + {inter_terms}'
            model_fit = smf.ols(formula, data=analysis_df).fit()
            anova_table = sm.stats.anova_lm(model_fit, typ=2).round(4)
            csv_path = os.path.join(RESULTS_FOLDER, 'anova_table.csv')
            anova_table.to_csv(csv_path)
            for line in anova_table.to_string().split('\n'):
                summary.write('  ' + line)

            if 'Residual' in anova_table.index:
                resid_ss = anova_table.loc['Residual', 'sum_sq']
                resid_df = anova_table.loc['Residual', 'df']
                for term in anova_table.index:
                    if term == 'Residual':
                        continue
                    try:
                        ss_term = anova_table.loc[term, 'sum_sq']
                        df_term = anova_table.loc[term, 'df']
                        F_val = anova_table.loc[term, 'F']
                        eta2 = ss_term / (ss_term + resid_ss)
                        try:
                            nc_lower = scipy.stats.ncf.ppf(0.025, df_term, resid_df, 0)
                            nc_upper = scipy.stats.ncf.ppf(0.975, df_term, resid_df, 0)
                            eta_low = nc_lower / (nc_lower + df_term + resid_df + 1)
                            eta_high = nc_upper / (nc_upper + df_term + resid_df + 1)
                        except Exception:
                            eta_low = np.nan
                            eta_high = np.nan
                        summary.write(f"  partial_eta2_{term} = {eta2:.4f} [95% CI {eta_low:.4f}-{eta_high:.4f}]")
                        metrics_table_records.append(("ANOVA_eta2", term, eta2))
                        metrics_table_records.append(("ANOVA_eta2_low", term, eta_low))
                        metrics_table_records.append(("ANOVA_eta2_high", term, eta_high))
                    except Exception:
                        pass

            cell_stats = analysis_df.copy()
            cell_stats['anova_pred'] = model_fit.fittedvalues
            cell_means = cell_stats.groupby(list(param_grid.keys())).agg(
                obs=('optimal_g_mse','mean'),
                pred=('anova_pred','mean'),
                n=('optimal_g_mse','size')
            )
            cell_means['resid'] = cell_means['obs'] - cell_means['pred']
            top_resid = cell_means.reindex(cell_means['resid'].abs().sort_values(ascending=False).index).head(10)
            summary.write("  Top 10 cells by absolute residual (observed - predicted):")
            for _, row in top_resid.reset_index().iterrows():
                cell_desc = ', '.join([f"{f}={row[f]}" for f in param_grid.keys()])
                summary.write(f"    {cell_desc}: resid={row['resid']:.4f} (n={row['n']})")
                metrics_table_records.append(("ANOVA_resid", cell_desc, row['resid']))
        except Exception as e_anova:
            summary.write(f"  Could not compute ANOVA: {e_anova}")
    
    if analysis_df.empty or len(analysis_df) < 10: # Adjusted minimum for robustness
        summary.write("\nNot enough valid data points after simulation and NaN removal to proceed with model building.")
    else:
        summary.write(f"\n--- Visualizations of Input Data ---")
        target_variable = 'optimal_g_mse'
        original_features = list(param_grid.keys())
        engineered_features = ['input_snr_proxy', 'saturation_headroom_ratio', 'gain_noise_sensitivity_proxy']
        all_features_for_model = [f for f in original_features + engineered_features if f in analysis_df.columns]
        
        # Scatter plots
        num_orig_features = len(original_features)
        scatter_cols = 3
        scatter_rows = (num_orig_features + scatter_cols - 1) // scatter_cols # Calculate rows needed
        fig_scatter, axes_scatter = plt.subplots(scatter_rows, scatter_cols, figsize=(18, 4 * scatter_rows))
        axes_scatter = axes_scatter.flatten()
        
        for idx, feat_name in enumerate(original_features):
            if feat_name in analysis_df.columns:
                axes_scatter[idx].scatter(analysis_df[feat_name], analysis_df[target_variable], alpha=0.5, s=15)
                axes_scatter[idx].set_xlabel(feat_name); axes_scatter[idx].set_ylabel(target_variable)
                axes_scatter[idx].set_title(f'{target_variable} vs. {feat_name}', fontsize=20); axes_scatter[idx].grid(True, alpha=0.3)
        for i in range(num_orig_features, len(axes_scatter)): fig_scatter.delaxes(axes_scatter[i]) # Remove unused subplots
        plt.tight_layout(); scatter_path = os.path.join(RESULTS_FOLDER, "optimal_g_vs_original_features.png"); plt.savefig(scatter_path); plt.close(fig_scatter)
        summary.write(f"Scatter plots of optimal_g vs original features saved to {scatter_path}")
        try:
            summary.write(
                f"FIG_scatter_all: file={scatter_path}, x=multiple_features, y={target_variable}, n={len(analysis_df)}"
            )
        except Exception:
            pass
        try:
            x_min = float(min(analysis_df[f].min() for f in original_features))
            x_max = float(max(analysis_df[f].max() for f in original_features))
            y_min = float(analysis_df[target_variable].min())
            y_max = float(analysis_df[target_variable].max())
            summary.write(
                f"FIG_scatter_axes: x=[{x_min:.3g},{x_max:.3g}], y=[{y_min:.3g},{y_max:.3g}]; n_points={len(analysis_df)}"
            )
        except Exception:
            pass

        # Include quartiles in the percentiles list so that '25%' and '75%'
        # keys are present when we access them below. Omitting them results in
        # a KeyError if pandas does not automatically include these values.
        target_desc = analysis_df[target_variable].describe(
            percentiles=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
        )
        deciles = [target_desc[f"{int(p*100)}%"] for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
        summary.write(
            f"Target '{target_variable}' stats: mean={target_desc['mean']:.4f} ± {target_desc['std']:.4f}, "
            f"median={target_desc['50%']:.4f}, IQR=[{target_desc['25%']:.4f}, {target_desc['75%']:.4f}], "
            f"deciles={deciles}"
        )
        try:
            plateau_mask = analysis_df[target_variable] >= 0.999
            plateau_pct = plateau_mask.mean() * 100
            summary.write(f"  {plateau_pct:.1f}% of all cases sit on the unit-MSE plateau.")
            value_counts = analysis_df[target_variable].value_counts().sort_index()
            freq_str = ", ".join([f"{idx:.3g}:{cnt}" for idx, cnt in value_counts.items()])
            summary.write(f"  {target_variable} frequency table: {freq_str}")
            analysis_df['is_plateau'] = plateau_mask
            summary.write("  Plateau rate by factor:")
            for factor in param_grid.keys():
                if factor in analysis_df.columns:
                    rates = analysis_df.groupby(factor)['is_plateau'].mean()
                    rate_str = ", ".join([f"{lvl}:{pct*100:.1f}%" for lvl, pct in rates.items()])
                    summary.write(f"    {factor}: {rate_str}")
            metrics_table_records.append(("Target", "Plateau_pct", plateau_pct))
        except Exception as e_plateau:
            summary.write(f"  Could not compute plateau stats: {e_plateau}")

        # --- Textual Summary for Scatter Plots ---
        summary.write("\n  Textual Summary of Scatter Plots (optimal_g_mse vs. Original Features):")
        summary.write(f"    Target variable: '{target_variable}' (Overall range: {analysis_df[target_variable].min():.3f} to {analysis_df[target_variable].max():.3f}, Mean: {analysis_df[target_variable].mean():.3f})")
        for feat_name in original_features:
            if feat_name in analysis_df.columns:
                feature_data = analysis_df[feat_name]
                target_data = analysis_df[target_variable]
                correlation = np.nan
                try:
                    if feature_data.nunique() > 1 and target_data.nunique() > 1: # Correlation needs variance
                         correlation = feature_data.corr(target_data)
                except Exception as e:
                    summary.write(f"      Could not calculate correlation for {feat_name}: {e}")

                summary.write(f"    - vs. '{feat_name}':")
                summary.write(f"        '{feat_name}' range: {feature_data.min():.3g} to {feature_data.max():.3g} (Mean: {feature_data.mean():.3g})")
                if not np.isnan(correlation):
                    summary.write(f"        Pearson correlation with '{target_variable}': {correlation:.3f}")
                    if abs(correlation) > 0.7: trend_desc = "strong"
                    elif abs(correlation) > 0.4: trend_desc = "moderate"
                    elif abs(correlation) > 0.1: trend_desc = "weak"
                    else: trend_desc = "very weak or no clear linear"
                    
                    if correlation > 0: trend_desc += " positive"
                    elif correlation < 0: trend_desc += " negative"
                    else: trend_desc = "no linear" # if correlation is exactly 0
                    summary.write(f"        This suggests a {trend_desc} relationship. Visual inspection is key for non-linearities.")
                else:
                    summary.write(f"        The plot for '{feat_name}' should be visually inspected to understand its relationship with '{target_variable}'.")

                try:
                    spear_corr, spear_p = scipy.stats.spearmanr(feature_data, target_data)
                    lr_res = scipy.stats.linregress(feature_data, target_data)
                    # Fit quadratic model for AIC/BIC comparison
                    X_lin = np.column_stack([np.ones(len(feature_data)), feature_data])
                    beta_lin, _, _, _ = np.linalg.lstsq(X_lin, target_data, rcond=None)
                    resid_lin = target_data - X_lin.dot(beta_lin)
                    sse_lin = np.sum(resid_lin**2)
                    k_lin = 2
                    aic_lin = len(target_data) * np.log(sse_lin/len(target_data)) + 2*k_lin
                    bic_lin = len(target_data) * np.log(sse_lin/len(target_data)) + k_lin*np.log(len(target_data))

                    X_quad = np.column_stack([np.ones(len(feature_data)), feature_data, feature_data**2])
                    beta_quad, _, _, _ = np.linalg.lstsq(X_quad, target_data, rcond=None)
                    resid_quad = target_data - X_quad.dot(beta_quad)
                    sse_quad = np.sum(resid_quad**2)
                    k_quad = 3
                    aic_quad = len(target_data) * np.log(sse_quad/len(target_data)) + 2*k_quad
                    bic_quad = len(target_data) * np.log(sse_quad/len(target_data)) + k_quad*np.log(len(target_data))
                    monotonic_flag = abs(spear_corr) > 0.5
                    summary.write(
                        f"        Spearman ρ={spear_corr:.3f} (p={spear_p:.3g}); monotonic≈{monotonic_flag}; "
                        f"OLS slope={lr_res.slope:.3f} ± {lr_res.stderr:.3f} (p={lr_res.pvalue:.3g}); "
                        f"ΔAIC={aic_quad-aic_lin:.3f}, ΔBIC={bic_quad-bic_lin:.3f} for quadratic term"
                    )
                    metrics_table_records.append((f"Scatter_{feat_name}", "Monotonic", monotonic_flag))
                    if aic_quad < aic_lin:
                        try:
                            coeffs, cov = np.polyfit(feature_data, target_data, 2, cov=True)
                            beta2 = coeffs[0]
                            se2 = np.sqrt(cov[0,0])
                            t_val = beta2 / se2 if se2 > 0 else np.nan
                            p_val = 2*(1-scipy.stats.t.cdf(abs(t_val), df=len(feature_data)-3)) if se2>0 else np.nan
                            summary.write(
                                f"        Quadratic term significant (β₂ = {beta2:.4f} ± {se2:.4f}, p={p_val:.3g})"
                            )
                        except Exception as e_quadfit:
                            summary.write(f"        Could not compute quadratic term significance: {e_quadfit}")
                    slope_ci_low = lr_res.slope - 1.96 * lr_res.stderr
                    slope_ci_high = lr_res.slope + 1.96 * lr_res.stderr
                    kend_tau, kend_p = scipy.stats.kendalltau(feature_data, target_data)
                    summary.write(
                        f"        OLS slope 95% CI: [{slope_ci_low:.3f}, {slope_ci_high:.3f}], n={len(feature_data)}"
                    )
                    summary.write(
                        f"        Kendall τ={kend_tau:.3f} (p={kend_p:.3g})"
                    )
                    try:
                        loess = sm.nonparametric.lowess(target_data, feature_data, frac=0.3, return_sorted=False)
                        q_bins = pd.qcut(feature_data, 10, labels=False, duplicates='drop')
                        slope_by_decile = []
                        for dec in range(10):
                            mask = q_bins == dec
                            if mask.sum() >= 2:
                                slope = np.polyfit(feature_data[mask], loess[mask], 1)[0]
                            else:
                                slope = np.nan
                            slope_by_decile.append(float(slope))
                        summary.write(f"        LOWESS slope by decile: {slope_by_decile}")
                        metrics_table_records.append((f"Scatter_{feat_name}", "LOWESS_slopes", slope_by_decile))
                    except Exception as e_low:
                        summary.write(f"        Could not compute LOWESS slopes: {e_low}")
                    try:
                        data_mat = np.vstack([feature_data, target_data]).T
                        cov = np.cov(data_mat, rowvar=False)
                        inv_cov = np.linalg.inv(cov)
                        mean_vec = data_mat.mean(axis=0)
                        diffs = data_mat - mean_vec
                        m_dist_sq = np.sum(diffs @ inv_cov * diffs, axis=1)
                        out_count = int(np.sum(m_dist_sq > 9.0))
                        summary.write(
                            f"        Number of points outside ±3 SD ellipse: {out_count} / {len(feature_data)}"
                        )
                        if feat_name == 'sigma_in_sq' and STATSMODELS_AVAILABLE:
                            try:
                                bp = het_breuschpagan(target_data, sm.add_constant(feature_data))
                                bp_p_raw = bp[1]
                                var_by = analysis_df.groupby('sigma_in_sq')[target_variable].var()
                                scale_ratio = (var_by.max() / var_by.min()) if var_by.min() > 0 else np.inf
                                summary.write(
                                    f"        Breusch-Pagan p (raw)={bp_p_raw:.3g}; variance scale max/min={scale_ratio:.1f}"
                                )
                                metrics_table_records.append(("Scatter_sigma_in_sq", "BP_raw_p", bp_p_raw))
                                metrics_table_records.append(("Scatter_sigma_in_sq", "VarScale", scale_ratio))
                            except Exception as e_bp:
                                summary.write(f"        Could not compute heteroscedasticity metrics: {e_bp}")
                    except Exception as e_out:
                        summary.write(f"        Could not compute outlier count: {e_out}")
                except Exception as e_stats:
                    summary.write(f"        Could not compute extended stats for {feat_name}: {e_stats}")

                # --- Additional quantitative cue: quadratic fit for curvature ---
                quad_r2 = np.nan
                quad_coeff = (np.nan, np.nan, np.nan)
                try:
                    if feature_data.nunique() > 2 and target_data.nunique() > 2:
                        coeffs = np.polyfit(feature_data, target_data, 2)
                        pred = np.polyval(coeffs, feature_data)
                        ss_res = np.sum((target_data - pred) ** 2)
                        ss_tot = np.sum((target_data - target_data.mean()) ** 2)
                        quad_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                        quad_coeff = coeffs
                except Exception as e:
                    summary.write(f"        Could not compute quadratic fit for {feat_name}: {e}")
                if not np.isnan(quad_r2):
                    summary.write(
                        f"        Quadratic fit R^2: {quad_r2:.3f}; second-order coefficient: {quad_coeff[0]:.4f}"
                    )

                # Additional distribution and influence metrics
                try:
                    dip_val, dip_p = approx_hartigans_diptest(feature_data)
                    n_peaks = count_kde_peaks(feature_data)
                    q1 = target_data.quantile(0.25)
                    q3 = target_data.quantile(0.75)
                    iqr = q3 - q1
                    fence_low = q1 - 1.5 * iqr
                    fence_high = q3 + 1.5 * iqr
                    out_mask = (target_data < fence_low) | (target_data > fence_high)
                    out_frac = float(out_mask.mean())
                    out_count = int(out_mask.sum())
                    cd95, high_count = cooks_distance_95th(feature_data, target_data)
                    summary.write(
                        f"        Dip p={dip_p:.3f}; KDE peaks={n_peaks}; % outside 1.5×IQR={out_frac*100:.1f}% (n={out_count}); Cook's D 95th pct={cd95:.4f}"
                    )
                    metrics_table_records.append((f"Scatter_{feat_name}", "Dip_p", dip_p))
                    metrics_table_records.append((f"Scatter_{feat_name}", "Peaks", n_peaks))
                    metrics_table_records.append((f"Scatter_{feat_name}", "Outlier_pct", out_frac*100))
                    metrics_table_records.append((f"Scatter_{feat_name}", "Outlier_n", out_count))
                    metrics_table_records.append((f"Scatter_{feat_name}", "CooksD95", cd95))
                except Exception as e_extra:
                    summary.write(f"        Could not compute extra scatter stats: {e_extra}")

                try:
                    dcor_val = distance_correlation(feature_data, target_data)
                    hull_area, ell_w1, ell_w2, ell_theta = point_cloud_metrics(feature_data, target_data)
                    summary.write(
                        f"        dCor={dcor_val:.3f}; hull area={hull_area:.4f}; "
                        f"95% ellipse widths≈({ell_w1:.3g},{ell_w2:.3g}), θ≈{ell_theta:.1f}°"
                    )
                    metrics_table_records.append((f"Scatter_{feat_name}", "dCor", dcor_val))
                    metrics_table_records.append((f"Scatter_{feat_name}", "HullArea", hull_area))
                    metrics_table_records.append((f"Scatter_{feat_name}", "EllW1", ell_w1))
                    metrics_table_records.append((f"Scatter_{feat_name}", "EllW2", ell_w2))
                    metrics_table_records.append((f"Scatter_{feat_name}", "EllTheta", ell_theta))
                except Exception:
                    pass
        
        # Pair plot
        pair_plot_vars = [f for f in original_features + [target_variable] if f in analysis_df.columns]
        sample_size_pairplot = min(len(analysis_df), 200) 
        if sample_size_pairplot > 1 and len(pair_plot_vars) > 1: # Need at least 2 data points and 2 vars
            sns.pairplot(analysis_df[pair_plot_vars].sample(sample_size_pairplot, random_state=42, replace=False), kind='scatter', diag_kind='kde', corner=True, plot_kws={'alpha':0.4, 's':15})
            plt.suptitle('Pair Plot of Original Features and Optimal g_mse (Sampled)', y=1.02, fontsize=24); pairplot_path = os.path.join(RESULTS_FOLDER, "pairplot_original_features.png"); plt.savefig(pairplot_path); plt.close()
            summary.write(f"\nPair plot saved to {pairplot_path}")
            try:
                summary.write(
                    f"FIG_pairplot_original: file={pairplot_path}, vars={','.join(pair_plot_vars)}, n={sample_size_pairplot}"
                )
            except Exception:
                pass
            try:
                all_mins = [analysis_df[v].min() for v in pair_plot_vars]
                all_maxs = [analysis_df[v].max() for v in pair_plot_vars]
                x_min = float(min(all_mins))
                x_max = float(max(all_maxs))
                summary.write(
                    f"FIG_pairplot_axes: x=[{x_min:.3g},{x_max:.3g}], y=[{x_min:.3g},{x_max:.3g}]; n_points={sample_size_pairplot}"
                )
            except Exception:
                pass

            # --- Textual Summary for Pair Plot ---
            summary.write("\n  Textual Summary of Pair Plot (Original Features and Optimal g_mse):")
            summary.write(f"    The pair plot visualizes pairwise relationships and distributions for: {', '.join(pair_plot_vars)}.")
            if len(analysis_df) > sample_size_pairplot:
                 summary.write(f"    (Based on a random sample of {sample_size_pairplot} data points)")

            summary.write("    Distribution Summary (from diagonal KDE plots):")
            for var_name in pair_plot_vars:
                if var_name in analysis_df.columns:
                    data_series = analysis_df[var_name]
                    desc_stats = data_series.describe()
                    q1 = data_series.quantile(0.25)
                    q3 = data_series.quantile(0.75)
                    skewness = data_series.skew() if data_series.nunique() > 1 else 0
                    kurt = data_series.kurtosis() if data_series.nunique() > 1 else 0
                    
                    summary.write(
                        f"    - '{var_name}': Mean={desc_stats['mean']:.3g}, Std={desc_stats['std']:.3g}, Min={desc_stats['min']:.3g}, Q1={q1:.3g}, Median={desc_stats['50%']:.3g}, Q3={q3:.3g}, Max={desc_stats['max']:.3g}, Skewness={skewness:.2f}, Kurtosis={kurt:.2f}."
                    )
                    if data_series.nunique() > 1:
                        if skewness > 0.5: skew_desc = "positively skewed"
                        elif skewness < -0.5: skew_desc = "negatively skewed"
                        else: skew_desc = "approximately symmetric"
                        summary.write(f"      The distribution of '{var_name}' appears {skew_desc}.")
                    else:
                        summary.write(f"      The distribution of '{var_name}' has no variance (constant value).")

                    bw = silverman_bandwidth(data_series)
                    summary.write(f"      Silverman's bandwidth h≈{bw:.4g}")
                    metrics_table_records.append((f"Pair_{var_name}", "Silverman_h", bw))
                    dip_p = np.nan
                    kde_peaks = np.nan
                    try:
                        dip_val, dip_p = approx_hartigans_diptest(data_series)
                        kde_peaks = count_kde_peaks(data_series)
                    except Exception as e_dip:
                        summary.write(f"      Could not compute modality metrics: {e_dip}")
                    summary.write(
                        f"      Dip p={dip_p:.3f}; KDE peaks={kde_peaks}"
                    )
                    metrics_table_records.append((f"Pair_{var_name}", "Dip_p", dip_p))
                    metrics_table_records.append((f"Pair_{var_name}", "KDE_peaks", kde_peaks))


            summary.write("\n    Pairwise Relationship Summary (from off-diagonal scatter plots):")
            if len(pair_plot_vars) > 1:
                # Calculate correlation on the full dataset for accuracy, not just sample
                corr_matrix = analysis_df[pair_plot_vars].corr()
                summary.write("      Notable Pearson Correlations (absolute value > 0.5):")
                reported_pairs = set()
                found_strong_corr = False
                for i in range(len(pair_plot_vars)):
                    for j in range(i + 1, len(pair_plot_vars)):
                        var1 = pair_plot_vars[i]
                        var2 = pair_plot_vars[j]
                        correlation = corr_matrix.loc[var1, var2]
                        dip_p_pair = np.nan
                        peaks_pair = np.nan
                        try:
                            pair_data = analysis_df[[var1, var2]].dropna()
                            if len(pair_data) >= 3:
                                centered = pair_data - pair_data.mean()
                                u, s, vh = np.linalg.svd(centered, full_matrices=False)
                                pc1 = centered @ vh[0]
                                dip_val_pair, dip_p_pair = approx_hartigans_diptest(pc1)
                                peaks_pair = count_kde_peaks(pc1)
                                dcor_pair = distance_correlation(pair_data[var1], pair_data[var2])
                        except Exception as e_pair:
                            summary.write(f"        Could not compute dip/KDE for {var1}-{var2}: {e_pair}")
                            dcor_pair = np.nan
                        if abs(correlation) >= 0.5:
                            summary.write(f"      - '{var1}' and '{var2}': {correlation:.3f}, dCor={dcor_pair:.3f}")
                            reported_pairs.add(tuple(sorted((var1, var2))))
                            found_strong_corr = True
                        else:
                            summary.write(f"      - {var1} vs {var2}: ρ={correlation:.3f}, dCor={dcor_pair:.3f}")
                        metrics_table_records.append((f"Pair_{var1}_{var2}", "Dip_p", dip_p_pair))
                        metrics_table_records.append((f"Pair_{var1}_{var2}", "KDE_peaks", peaks_pair))
                        metrics_table_records.append((f"Pair_{var1}_{var2}", "dCor", dcor_pair))
                if not found_strong_corr:
                    summary.write("      - No pairs found with absolute correlation >= 0.5 among these variables.")
                summary.write("      Visual inspection of scatter plots is crucial for identifying non-linear relationships or patterns not captured by Pearson correlation.")

                corr_csv = corr_matrix.to_csv(index=True)
                summary.write("\n      Full Pearson correlation matrix:")
                for line in corr_csv.strip().split("\n"):
                    summary.write(f"        {line}")

                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                abs_pairs = upper_tri.abs().unstack().dropna().sort_values(ascending=False)
                top_pairs = abs_pairs.head(5)
                summary.write("      Top 5 highest |ρ| pairs:")
                for (v1, v2), val in top_pairs.items():
                    summary.write(f"        {v1} vs {v2}: {val:.3f}")
            else:
                summary.write("      Not enough variables for pairwise comparisons.")
        else:
            summary.write("\nSkipping pair plot and its textual summary due to insufficient data or features.")


        summary.write(f"\n--- Heuristic Model Development ---")
        X = analysis_df[all_features_for_model]
        y = analysis_df['optimal_g_mse']
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        summary.write(f"Data split: Train/Validation set size: {len(X_train_val)}, Test set size: {len(X_test)}")

        models_to_evaluate = {
            "Linear Regression": (lambda: Pipeline([("scaler", StandardScaler()), ("lin_reg", LinearRegression())]), {}),
            "Polynomial Regression (Deg=2)": (lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
                ("lin_reg", LinearRegression())
            ]), {}),
            "Random Forest": (RandomForestRegressor, {
                'n_estimators': [50, 100], 'max_depth': [None, 10], # Reduced for speed
                'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]
            }),
            "Gradient Boosting": (GradientBoostingRegressor, {
                'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], # Reduced for speed
                'max_depth': [3, 5], 'subsample': [0.7, 1.0]
            })
        }
        
        trained_models = {}
        model_cv_performance = {}
        model_cv_metrics = {}

        summary.write("\n--- Model Training, Tuning, and Cross-Validation ---")
        for name, (model_constructor, param_dist) in models_to_evaluate.items():
            model, cv_mse, best_params, cv_metrics = train_evaluate_tune_model(
                model_constructor, X_train_val, y_train_val, name, summary,
                param_dist=param_dist, n_iter_search=5, # Reduced n_iter for speed
                feature_names_list=all_features_for_model,
                cv_folds=10
            )
            trained_models[name] = model
            model_cv_performance[name] = cv_mse
            if cv_metrics is not None:
                model_cv_metrics[name] = cv_metrics

        summary.write("\n--- Systematic Model Comparison (CV MSE on Train/Val set) ---")
        if model_cv_performance and not all(np.isnan(v) for v in model_cv_performance.values()):
            performance_summary = pd.Series(model_cv_performance).sort_values()
            summary.write(performance_summary.to_string())

            for m_name, metrics in model_cv_metrics.items():
                try:
                    mse_scores = metrics.get('mse', [])
                    mae_scores = metrics.get('mae', [])
                    r2_scores = metrics.get('r2', [])
                    summary.write(
                        f"    {m_name} CV fold metrics -> MSE mean±SD: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}; "
                        f"RMSE mean±SD: {np.mean(np.sqrt(mse_scores)):.4f} ± {np.std(np.sqrt(mse_scores)):.4f}; "
                        f"MAE mean±SD: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}; "
                        f"R² mean±SD: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}"
                    )
                except Exception as e_metrics:
                    summary.write(f"    Could not summarize CV fold metrics for {m_name}: {e_metrics}")
            
            # Filter out NaN MSEs before finding idxmin
            valid_performance = performance_summary.dropna()
            if not valid_performance.empty:
                best_model_name = valid_performance.idxmin()
                final_best_model = trained_models[best_model_name]
                summary.write(f"\nSelected best model based on CV MSE: {best_model_name} (CV MSE: {valid_performance.min():.4f})")

                if final_best_model and not X_test.empty:
                    y_pred_test = final_best_model.predict(X_test)
                    test_set_mse = model_evaluation_mse(y_test, y_pred_test)
                    test_set_rmse = np.sqrt(test_set_mse)
                    test_set_mae = np.mean(np.abs(y_test - y_pred_test))
                    ss_res = np.sum((y_test - y_pred_test) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    test_set_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                    summary.write(
                        f"Performance of '{best_model_name}' on Hold-Out Test Set: "
                        f"MSE={test_set_mse:.4f}, RMSE={test_set_rmse:.4f}, "
                        f"MAE={test_set_mae:.4f}, R²={test_set_r2:.4f}"
                    )

                    try:
                        dummy = DummyRegressor(strategy='mean')
                        dummy.fit(X_train_val, y_train_val)
                        baseline_pred = dummy.predict(X_test)
                        baseline_mse = model_evaluation_mse(y_test, baseline_pred)
                        summary.write(
                            f"  DummyRegressor baseline MSE on test set: {baseline_mse:.4f}"
                        )
                        metrics_table_records.append(("Baseline", "Dummy_MSE", baseline_mse))
                    except Exception as e_dummy:
                        summary.write(f"  Could not compute dummy baseline: {e_dummy}")

                    # Bootstrap CI for RMSE
                    try:
                        n_boot = 1000
                        rng = np.random.default_rng(42)
                        boot_rmses = []
                        boot_mses = []
                        boot_maes = []
                        for _ in range(n_boot):
                            idx = rng.integers(0, len(y_test), len(y_test))
                            boot_rmse = np.sqrt(np.mean((y_test.iloc[idx].values - y_pred_test[idx]) ** 2))
                            boot_mse = np.mean((y_test.iloc[idx].values - y_pred_test[idx]) ** 2)
                            boot_mae = np.mean(np.abs(y_test.iloc[idx].values - y_pred_test[idx]))
                            boot_rmses.append(boot_rmse)
                            boot_mses.append(boot_mse)
                            boot_maes.append(boot_mae)
                        ci_low, ci_high = np.percentile(boot_rmses, [2.5, 97.5])
                        ci_mse_low, ci_mse_high = np.percentile(boot_mses, [2.5, 97.5])
                        ci_mae_low, ci_mae_high = np.percentile(boot_maes, [2.5, 97.5])
                        summary.write(
                            f"  Bootstrap 95% CI for test RMSE: [{ci_low:.4f}, {ci_high:.4f}]"
                        )
                        summary.write(
                            f"  Bootstrap 95% CI for test MSE: [{ci_mse_low:.4f}, {ci_mse_high:.4f}]"
                        )
                        summary.write(
                            f"  Bootstrap 95% CI for test MAE: [{ci_mae_low:.4f}, {ci_mae_high:.4f}]"
                        )
                    except Exception as e_boot:
                        summary.write(f"  Could not compute bootstrap CI for RMSE: {e_boot}")

                    # Residual diagnostics
                    try:
                        residuals = y_test - y_pred_test
                        res_mean = np.mean(residuals)
                        res_std = np.std(residuals)
                        res_skew = scipy.stats.skew(residuals)
                        res_kurt = scipy.stats.kurtosis(residuals)
                        dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
                        # Split fitted values into deciles for Levene's test
                        fitted_deciles = pd.qcut(y_pred_test, 10, duplicates='drop')
                        groups = [residuals[fitted_deciles == level] for level in fitted_deciles.unique()]
                        levene_p = scipy.stats.levene(*groups).pvalue if len(groups) > 1 else np.nan
                        summary.write(
                            f"  Residuals: mean={res_mean:.4f}, SD={res_std:.4f}, skew={res_skew:.2f}, "
                            f"kurtosis={res_kurt:.2f}, Durbin-Watson={dw_stat:.3f}, Levene p={levene_p:.3f}"
                        )
                        try:
                            shapiro_p = scipy.stats.shapiro(residuals).pvalue
                            if STATSMODELS_AVAILABLE:
                                bp_test = het_breuschpagan(residuals, np.column_stack([np.ones(len(y_pred_test)), y_pred_test]))
                                bp_p = bp_test[1]
                                bp_r2 = bp_test[2]  # R^2 of auxiliary regression
                            else:
                                bp_p = np.nan
                                bp_r2 = np.nan
                            summary.write(
                                f"    Shapiro-Wilk p={shapiro_p:.3f}; Breusch-Pagan p={bp_p:.3f}"
                            )
                            metrics_table_records.append(("Residuals", "BP_R2", bp_r2))
                            try:
                                cd95, cdcount = cooks_distance_95th(y_pred_test, y_test.values)
                                summary.write(
                                    f"    Cook's D 95th pct={cd95:.4f}, high leverage count={cdcount}"
                                )
                                metrics_table_records.append(("Residuals", "CooksD95", cd95))
                                metrics_table_records.append(("Residuals", "HighLeverage", cdcount))
                            except Exception as e_cook:
                                summary.write(f"    Could not compute Cook's distance: {e_cook}")
                        except Exception as e_diag:
                            summary.write(f"    Could not compute Shapiro/Breusch-Pagan: {e_diag}")
                    except Exception as e_res:
                        summary.write(f"  Could not compute residual diagnostics: {e_res}")

                    if isinstance(final_best_model, (RandomForestRegressor, GradientBoostingRegressor)):
                        summary.write("  Computing permutation feature importance...")
                        try:
                            perm = permutation_importance(
                                final_best_model, X_test, y_test, n_repeats=30,
                                random_state=42, scoring='neg_mean_squared_error'
                            )
                            importances_mean = perm.importances_mean
                            importances_std = perm.importances_std
                            total_abs = np.sum(np.abs(importances_mean))
                            for fname, mean_imp, std_imp in zip(X_test.columns, importances_mean, importances_std):
                                rel = (abs(mean_imp) / total_abs * 100) if total_abs != 0 else np.nan
                                summary.write(f"    {fname}: {mean_imp:.5f} ± {std_imp:.5f} ({rel:.1f}% of total)")
                                metrics_table_records.append(("PermImp", fname, rel))
                        except Exception as e_perm:
                            summary.write(f"  Error computing permutation importance: {e_perm}")
                    
                    model_save_path = os.path.join(RESULTS_FOLDER, f"best_heuristic_model_{best_model_name.replace(' ', '_')}.joblib")
                    joblib.dump(final_best_model, model_save_path)
                    summary.write(f"Saved best model to {model_save_path}")

                    if isinstance(final_best_model, RandomForestRegressor):
                        try:
                            preds_cal, obs_cal = calibration_curve_regression(y_test.values, y_pred_test)
                            calib_df = pd.DataFrame({'pred_bin': preds_cal, 'observed': obs_cal})
                            calib_path = os.path.join(RESULTS_FOLDER, "calibration_curve.csv")
                            calib_df.to_csv(calib_path, index=False)
                            summary.write(f"Saved calibration curve to {calib_path}")
                            metrics_table_records.append(("Calibration", "n_bins", len(preds_cal)))
                        except Exception as e_cal:
                            summary.write(f"  Could not compute calibration curve: {e_cal}")

                    plot_heuristic_behavior(
                        final_best_model, analysis_df,
                        original_features, engineered_features,
                        RESULTS_FOLDER, summary
                    )
                    
                    if SHAP_AVAILABLE and isinstance(final_best_model, (RandomForestRegressor, GradientBoostingRegressor, Pipeline)):
                        summary.write("\n--- SHAP Value Analysis for Best Model ---")
                        try:
                            model_for_shap = final_best_model
                            # Use a smaller sample for SHAP if X_train_val is large, for speed
                            if len(X_train_val) > 200:
                                data_for_shap = shap.sample(X_train_val, 100, random_state=42)
                                summary.write(f"  Using a sample of 100 points from X_train_val for SHAP explainer due to dataset size.")
                            else:
                                data_for_shap = X_train_val
                            
                            # Background data for explainer (can be X_train_val or a sample)
                            # The data passed to explainer() is what gets explained.
                            # For TreeExplainer, masker is not strictly needed if data is pd.DataFrame.
                            # For KernelExplainer, background data is important.
                            # Let's use a smaller background for potentially slow explainers
                            background_data_shap = shap.sample(X_train_val, min(50, len(X_train_val)), random_state=42)
                            
                            explainer = shap.Explainer(model_for_shap, background_data_shap)
                            shap_values = explainer(data_for_shap, check_additivity=False) 

                            plt.figure() 
                            shap.summary_plot(shap_values, data_for_shap, show=False, plot_size=(10,8))
                            plt.tight_layout()
                            shap_summary_path = os.path.join(RESULTS_FOLDER, "shap_summary_plot.png")
                            # Save at higher DPI to avoid pixelation in book quality output
                            plt.savefig(
                                shap_summary_path,
                                dpi=500,
                                bbox_inches="tight",
                            )
                            plt.close()
                            summary.write(f"SHAP summary plot saved to {shap_summary_path}")
                            try:
                                summary.write(
                                    f"FIG_shap_summary: file={shap_summary_path}, n={shap_values.values.shape[0]}"
                                )
                            except Exception:
                                pass
                            try:
                                shap_min = float(np.min(shap_values.values))
                                shap_max = float(np.max(shap_values.values))
                                summary.write(
                                    f"FIG_shap_summary_axes: x=[{shap_min:.4f},{shap_max:.4f}], y=[0,{shap_values.values.shape[1]}]; n_points={shap_values.values.shape[0]}"
                                )
                            except Exception:
                                pass

                            # --- Textual Summary for SHAP Plot ---
                            summary.write("\n  Textual Summary of SHAP Value Analysis:")
                            s_values_arr = None
                            f_names_list = None
                            f_data_arr = None

                            if hasattr(shap_values, 'values') and hasattr(shap_values, 'feature_names') and hasattr(shap_values, 'data'):
                                s_values_arr = shap_values.values
                                f_names_list = shap_values.feature_names
                                f_data_arr = shap_values.data # This data corresponds to points for which SHAP values were computed
                                if not isinstance(f_names_list, list): f_names_list = list(f_names_list) # Ensure list
                            elif isinstance(shap_values, np.ndarray) and hasattr(data_for_shap, 'columns'): # Fallback for older SHAP or different explainer outputs
                                s_values_arr = shap_values
                                f_names_list = data_for_shap.columns.tolist()
                                f_data_arr = data_for_shap.values
                                summary.write("    (SHAP values interpreted as NumPy array, feature names from input data for SHAP.)")
                            
                            if s_values_arr is not None and f_names_list is not None and f_data_arr is not None:
                                if s_values_arr.ndim == 1: # Handle case of single output for some explainers
                                    s_values_arr = s_values_arr.reshape(-1, 1)
                                if f_data_arr.ndim == 1:
                                    f_data_arr = f_data_arr.reshape(-1,1)

                                if len(f_names_list) != s_values_arr.shape[1]:
                                    summary.write(f"    Warning: Mismatch in SHAP feature names ({len(f_names_list)}) and SHAP values dimension ({s_values_arr.shape[1]}). Textual summary might be partial or incorrect.")
                                    # Attempt to use all_features_for_model if lengths match that
                                    if len(all_features_for_model) == s_values_arr.shape[1]:
                                        f_names_list = all_features_for_model
                                        summary.write(f"    Using feature names from 'all_features_for_model'.")
                                    else: # Cannot reconcile, proceed with caution or skip part of summary
                                        summary.write(f"    Cannot reconcile feature names for SHAP textual summary. Skipping detailed impact description.")
                                        f_names_list = [f"Feature_{j}" for j in range(s_values_arr.shape[1])]


                                mean_abs_shap = np.abs(s_values_arr).mean(axis=0)
                                q25_abs_shap = np.percentile(np.abs(s_values_arr), 25, axis=0)
                                q75_abs_shap = np.percentile(np.abs(s_values_arr), 75, axis=0)
                                if len(f_names_list) == len(mean_abs_shap):
                                    feature_importance_shap = pd.Series(mean_abs_shap, index=f_names_list).sort_values(ascending=False)
                                    summary.write("    Global Feature Importance (based on mean absolute SHAP values):")
                                    top_sd_list = []
                                    for i, feat in enumerate(feature_importance_shap.index):
                                        idx = f_names_list.index(feat)
                                        mean_val = s_values_arr[:, idx].mean()
                                        std_val = s_values_arr[:, idx].std()
                                        feat_std = analysis_df[feat].std() if feat in analysis_df.columns else np.nan
                                        per_sd = (mean_val / feat_std) if feat_std not in [0, np.nan] else np.nan
                                        summary.write(
                                            f"      {i+1}. {feat}: mean={mean_abs_shap[idx]:.4f}, Q25={q25_abs_shap[idx]:.4f}, Q75={q75_abs_shap[idx]:.4f}; signed mean impact={mean_val:.4f} ± {std_val:.4f}; Δprediction per +1 SD ≈ {per_sd:.4f}"
                                        )
                                        frac_pos = float(np.mean(s_values_arr[:, idx] > 0))
                                        frac_neg = float(np.mean(s_values_arr[:, idx] < 0))
                                        iqr_shap = float(np.percentile(s_values_arr[:, idx], 75) - np.percentile(s_values_arr[:, idx], 25))
                                        summary.write(
                                            f"        Sign distribution: {frac_pos:.2f} >0, {frac_neg:.2f} <0; IQR={iqr_shap:.4f}"
                                        )
                                        median_shap = float(np.median(s_values_arr[:, idx]))
                                        skew_shap = float(scipy.stats.skew(s_values_arr[:, idx])) if s_values_arr.shape[0] > 2 else 0.0
                                        summary.write(
                                            f"        Median impact={median_shap:.4f}; skewness={skew_shap:.3f}"
                                        )
                                        min_shap = float(np.min(s_values_arr[:, idx]))
                                        max_shap = float(np.max(s_values_arr[:, idx]))
                                        frac_gt = float(np.mean(np.abs(s_values_arr[:, idx]) > 0.05)) * 100
                                        summary.write(
                                            f"        Min={min_shap:.4f}, Max={max_shap:.4f}, %>|0.05|={frac_gt:.1f}%"
                                        )
                                        metrics_table_records.append((f"SHAP_{feat}", "Frac_pos", frac_pos))
                                        metrics_table_records.append((f"SHAP_{feat}", "Frac_neg", frac_neg))
                                        metrics_table_records.append((f"SHAP_{feat}", "IQR", iqr_shap))
                                        metrics_table_records.append((f"SHAP_{feat}", "Median", median_shap))
                                        metrics_table_records.append((f"SHAP_{feat}", "Skew", skew_shap))
                                        metrics_table_records.append((f"SHAP_{feat}", "Min", min_shap))
                                        metrics_table_records.append((f"SHAP_{feat}", "Max", max_shap))
                                        metrics_table_records.append((f"SHAP_{feat}", "Pct_gt_0.05", frac_gt))
                                        if i < 3:
                                            top_sd_list.append(f"Δprediction per +1 SD {feat} = {per_sd:+.3f} g (SHAP)")

                                    cum_contrib = np.cumsum(feature_importance_shap.values) / np.sum(feature_importance_shap.values)
                                    cum_line = ", ".join([f"{i+1}:{c*100:.1f}%" for i, c in enumerate(cum_contrib)])
                                    summary.write(f"\n    Cumulative |SHAP| captured by top-k features: {cum_line}")
                                    if top_sd_list:
                                        summary.write("    " + "; ".join(top_sd_list))

                                    summary.write("\n    Impact of Top Features on Predictions (qualitative, from SHAP values):")
                                    num_top_features_to_describe = min(5, len(feature_importance_shap))
                                    for feat_name_shap in feature_importance_shap.index[:num_top_features_to_describe]:
                                        try:
                                            feat_idx_shap = f_names_list.index(feat_name_shap)
                                        except ValueError:
                                            summary.write(f"      Could not find index for feature '{feat_name_shap}' in SHAP feature names. Skipping.")
                                            continue
                                        
                                        # Ensure f_data_arr has same number of features as s_values_arr and f_names_list
                                        if f_data_arr.shape[1] != s_values_arr.shape[1]:
                                            summary.write(f"      Feature data array columns ({f_data_arr.shape[1]}) don't match SHAP values columns ({s_values_arr.shape[1]}). Cannot describe impact for {feat_name_shap}.")
                                            continue

                                        feature_column_original_values = f_data_arr[:, feat_idx_shap]
                                        feature_column_shap_values = s_values_arr[:, feat_idx_shap]
                                        
                                        direction_desc = "Impact direction complex or data insufficient for simple correlation."
                                        if pd.Series(feature_column_original_values).nunique() > 1 and pd.Series(feature_column_shap_values).nunique() > 1:
                                            try:
                                                corr_val_shap = np.corrcoef(feature_column_original_values, feature_column_shap_values)[0, 1]
                                                if np.isnan(corr_val_shap):
                                                    direction_desc = "Impact direction unclear (correlation is NaN)."
                                                elif abs(corr_val_shap) < 0.2:
                                                    direction_desc = f"Impact direction is complex or weak (correlation: {corr_val_shap:.2f}). High/low values do not consistently push prediction one way."
                                                elif corr_val_shap > 0:
                                                    direction_desc = f"Higher values of '{feat_name_shap}' tend to increase the prediction (correlation: {corr_val_shap:.2f})."
                                                else: # corr_val_shap < 0
                                                    direction_desc = f"Higher values of '{feat_name_shap}' tend to decrease the prediction (correlation: {corr_val_shap:.2f})."
                                            except Exception as e_corr:
                                                direction_desc = f"Could not calculate correlation for SHAP impact: {e_corr}"

                                        summary.write(f"    - {feat_name_shap}: {direction_desc}")

                                    if len(feature_importance_shap) > num_top_features_to_describe:
                                        summary.write(f"    (Impact descriptions shown for top {num_top_features_to_describe} features. Full importance list above.)")

                                    # SHAP interaction values for top features
                                    try:
                                        interaction_vals = shap.TreeExplainer(model_for_shap).shap_interaction_values(data_for_shap)
                                        if isinstance(interaction_vals, list):
                                            interaction_vals = interaction_vals[0]
                                        mean_inter = np.abs(interaction_vals).mean(axis=0)
                                        pairs = []
                                        for i_f1 in range(len(f_names_list)):
                                            for i_f2 in range(i_f1 + 1, len(f_names_list)):
                                                pairs.append((f_names_list[i_f1], f_names_list[i_f2], mean_inter[i_f1, i_f2]))
                                        top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
                                        summary.write("\n    Top SHAP Interaction Effects:")
                                        for f1, f2, val in top_pairs:
                                            summary.write(f"      {f1} × {f2}: mean |ϕ|={val:.4f}")
                                    except Exception as e_inter:
                                        summary.write(f"    Could not compute SHAP interaction values: {e_inter}")

                                    # SHAP value at selected quantiles
                                    summary.write("\n    SHAP value change across feature quantiles:")
                                    for feat_name_shap in feature_importance_shap.index[:num_top_features_to_describe]:
                                        try:
                                            feat_idx_shap = f_names_list.index(feat_name_shap)
                                            feat_col = f_data_arr[:, feat_idx_shap]
                                            shap_col = s_values_arr[:, feat_idx_shap]
                                            q10, q50, q90 = np.percentile(feat_col, [10, 50, 90])
                                            shap_at_q10 = np.mean(shap_col[feat_col <= q10])
                                            shap_at_q50 = np.mean(shap_col[(feat_col >= q50 - 1e-9) & (feat_col <= q50 + 1e-9)]) if np.any((feat_col >= q50 - 1e-9) & (feat_col <= q50 + 1e-9)) else np.mean(shap_col)
                                            shap_at_q90 = np.mean(shap_col[feat_col >= q90])
                                            summary.write(
                                                f"      {feat_name_shap}: SHAP at 10th={shap_at_q10:.4f}, 50th={shap_at_q50:.4f}, 90th={shap_at_q90:.4f}"
                                            )
                                        except Exception as e_q:
                                            summary.write(f"      Could not compute quantile SHAP for {feat_name_shap}: {e_q}")
                                else:
                                     summary.write("    Could not generate ranked SHAP feature importances due to name/value length mismatch.")
                            else:
                                summary.write("    Could not derive necessary SHAP arrays (values, feature names, or data) for full textual summary.")
                        except Exception as e_shap:
                            summary.write(f"  An error occurred during SHAP analysis or textual summary generation: {e_shap}")
                            import traceback
                            summary.write(f"  SHAP Error Traceback:\n{traceback.format_exc()}")
                else:
                    summary.write("\nCould not evaluate on test set or save best model as it was not successfully trained or test set is empty.")
            else:
                summary.write("\nBest model could not be determined from CV performance (all MSEs might be NaN).")
        else:
            summary.write("\nNo models were successfully evaluated or all returned NaN CV MSE.")

    total_script_time = time.time() - start_time_total
    summary.write(f"\n--- Experiment End ---")
    summary.write(f"Total script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes).")
    summary.write(f"\n--- Operating Conditions for Heuristic ---")
    summary.write(f"The developed heuristic model is trained on the parameter ranges defined in 'param_grid'.")
    summary.write(f"Its predictions are most reliable within these ranges. Extrapolation should be done with caution.")
    timestamp_end = time.strftime("%Y-%m-%d %H:%M:%S")
    summary.write(f"Timestamp: {timestamp_end}")

    if metrics_table_records:
        summary.write("\n--- Numerical Summary Table ---")
        summary.write("FIGURE, METRIC, VALUE")
        for fig, metric, val in metrics_table_records:
            summary.write(f"{fig}, {metric}, {val}")


# --- Main Execution Block ---
if __name__ == "__main__":
    summary_filepath = os.path.join(RESULTS_FOLDER, "experiment_5A_summary.txt")
    with SummaryWriter(summary_filepath, print_to_console=True) as summary:
        try:
            run_experiment(summary)
        except Exception as e:
            summary.write(f"\n!!!!!!!! AN UNHANDLED ERROR OCCURRED !!!!!!!!")
            summary.write(f"Error Type: {type(e).__name__}")
            summary.write(f"Error Message: {str(e)}")
            print(f"\n!!!!!!!! AN UNHANDLED ERROR OCCURRED !!!!!!!!")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
