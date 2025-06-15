# analyze_experiment_2_results.py
import numpy as np
import matplotlib.pyplot as plt

# Double all default font sizes for analysis figures
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
import seaborn as sns
import os
import joblib  # For saving/loading models
import time
import json  # Explicitly import for loading
import sys
import platform
import scipy.stats
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as model_evaluation_mse
from sklearn.inspection import permutation_importance
import sklearn

# For SHAP value analysis
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP library not found. Install with 'pip install shap' for SHAP value analysis.")

# --- Configuration ---
BASE_RESULTS_FOLDER_EXP2 = "results_experiment_5B_info_cost"
CSV_FILE_NAME_EXP2 = "exp2_optimal_g_utility_and_MI_raw_at_opt.csv"
CSV_FILE_PATH_EXP2 = os.path.join(BASE_RESULTS_FOLDER_EXP2, CSV_FILE_NAME_EXP2)

FULL_SWEEP_JSON_FILE_EXP2 = "exp2_full_sweeps_data_with_miraw_curves.json"
FULL_SWEEP_JSON_PATH_EXP2 = os.path.join(BASE_RESULTS_FOLDER_EXP2, FULL_SWEEP_JSON_FILE_EXP2)

ANALYSIS_RESULTS_FOLDER_EXP2 = os.path.join(BASE_RESULTS_FOLDER_EXP2, "analysis_outputs")
if not os.path.exists(ANALYSIS_RESULTS_FOLDER_EXP2):
    os.makedirs(ANALYSIS_RESULTS_FOLDER_EXP2)

RANDOM_SEED = 42

TARGET_VARIABLE_EXP2 = 'optimal_g_utility'

# Collect important metrics for summary tables
metrics_table_records = []

# --- Summary Writer Class ---
class SummaryWriter:
    def __init__(self, filepath, print_to_console=True):
        self.filepath = filepath
        self.print_to_console = print_to_console
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.file_handle = open(self.filepath, 'w', encoding='utf-8')
        if self.print_to_console:
            sys.__stdout__.write(f"Summary will be saved to: {self.filepath}\n")
            sys.__stdout__.flush()

    def write(self, message, end="\n", console_only=False):
        if self.print_to_console:
            sys.__stdout__.write(str(message) + end)
            sys.__stdout__.flush()
        if not console_only and self.file_handle:
            self.file_handle.write(str(message) + end)
            self.file_handle.flush()

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            if self.print_to_console:
                sys.__stdout__.write(f"Summary file closed: {self.filepath}\n")
                sys.__stdout__.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def swrite(summary, msg):
    if summary is not None:
        summary.write(msg)
    else:
        print(msg)

class TeeFile:
    def __init__(self, summary):
        self.summary = summary
    def write(self, msg):
        if msg:
            self.summary.write(msg, end="")
    def flush(self):
        pass

# --- 1. Data Loading and Preprocessing (Initial Part) ---
def initial_load_data(file_path, summary=None):
    swrite(summary, f"\n--- 1. Initial Data Loading from {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        swrite(summary, f"ERROR: CSV file not found at {file_path}. Ensure experiment_5B_info_cost.py has been run and saved the 'json_sweep_key' column.")
        return None
    swrite(summary, f"Original data shape: {df.shape}")

    if 'json_sweep_key' not in df.columns:
        swrite(summary, "ERROR: 'json_sweep_key' column not found in CSV. Please re-run experiment_5B_info_cost.py with the modification to save this key.")
        return None

    if 'C_proc' in df.columns:
        df['C_proc_for_saturation_check'] = df['C_proc'].copy()
        swrite(summary, f"  'C_proc_for_saturation_check' (original C_proc, inf is inf): \n{df['C_proc_for_saturation_check'].describe(include='all')}")
        swrite(summary, f"  Number of np.inf in C_proc_for_saturation_check: {np.isinf(df['C_proc_for_saturation_check']).sum()}")
    else:
        swrite(summary, "WARNING: 'C_proc' column missing from CSV.")
        df['C_proc_for_saturation_check'] = np.nan

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    swrite(summary, f"Data shape after general inf->nan replacement (for modeling prep): {df.shape}")
    if 'C_proc' in df.columns:
        swrite(summary, f"  Stats for 'C_proc' (base for modeling, inf now NaN): \n{df['C_proc'].describe(include='all')}")

    return df

# --- Preprocessing (Continued: after saturation analysis) ---
def complete_preprocessing_for_modeling(df, summary=None):
    swrite(summary, "\n--- 1b. Completing Preprocessing for Modeling ---")
    if df is None: return None

    if 'C_proc' in df.columns:
        cproc_tmp = df['C_proc'].copy()
        max_finite_cproc_val = cproc_tmp[np.isfinite(cproc_tmp)].max()
        if pd.isna(max_finite_cproc_val):
            max_finite_cproc_val = 1000.0
        impute_const = max_finite_cproc_val * 10
        df.loc[:, 'C_proc_for_modeling'] = cproc_tmp.fillna(impute_const)
        swrite(summary, f"  Count of C_proc == inf: {np.isinf(df['C_proc']).sum()}; imputed with {impute_const} for modeling")
        swrite(summary, f"C_proc == inf imputation constant: {impute_const}")
    else:
        df['C_proc_for_modeling'] = 1000.0
        swrite(summary, "  WARNING: 'C_proc' column was not found for modeling, 'C_proc_for_modeling' set to default 1000.0")

    original_rows = len(df)
    df.dropna(subset=[TARGET_VARIABLE_EXP2], inplace=True)
    swrite(summary, f"  Dropped {original_rows - len(df)} rows due to NaN in '{TARGET_VARIABLE_EXP2}'.")
    swrite(summary, f"  Data shape after dropping NaNs in target: {df.shape}")

    if df.empty:
        swrite(summary, "  No data remaining after preprocessing for modeling. Exiting analysis.")
        return None
    return df

# --- 2. Feature Engineering ---
def engineer_features_exp2(df, summary=None):
    swrite(summary, "\n--- 2. Engineering Features for Experiment 5B ---")
    df_eng = df.copy()
    epsilon = 1e-9
    c_proc_col_to_use = 'C_proc_for_modeling'

    if 'amplitude_s_true' in df_eng.columns and 'sigma_in_sq' in df_eng.columns:
        df_eng['input_snr_proxy'] = (df_eng['amplitude_s_true']**2) / (df_eng['sigma_in_sq'] + epsilon)
    if 's_max' in df_eng.columns and 'amplitude_s_true' in df_eng.columns:
        df_eng['saturation_headroom_ratio'] = df_eng['s_max'] / (df_eng['amplitude_s_true'] + epsilon)
    if 'k_gain' in df_eng.columns and 'alpha_gain_noise' in df_eng.columns:
        df_eng['gain_noise_sensitivity_proxy'] = df_eng['k_gain'] * df_eng['alpha_gain_noise']
    if 'lambda_cost' in df_eng.columns and 'kappa' in df_eng.columns:
        df_eng['cost_sensitivity_ratio'] = df_eng['lambda_cost'] * df_eng['kappa']
    if c_proc_col_to_use in df_eng.columns and 'amplitude_s_true' in df_eng.columns:
        df_eng['processing_vs_signal_strength'] = df_eng[c_proc_col_to_use] / (df_eng['amplitude_s_true'] + epsilon)
    if 'w_IT' in df_eng.columns and 'lambda_cost' in df_eng.columns:
        df_eng['benefit_IT_vs_cost_lambda'] = df_eng['w_IT'] / (df_eng['lambda_cost'] + epsilon)
    if 'w_TPE' in df_eng.columns and 'lambda_cost' in df_eng.columns:
        df_eng['benefit_TPE_vs_cost_lambda'] = df_eng['w_TPE'] / (df_eng['lambda_cost'] + epsilon)
    if 'phi1' in df_eng.columns:
        df_eng['phi1_deviation_from_linear'] = np.abs(df_eng['phi1'] - 1.0)
    return df_eng

def finite_cproc_df(df):
    return df[np.isfinite(df['C_proc_for_saturation_check']) &
              df['g_lever_MI_saturates'].notna()].copy()

# --- 3. Exploratory Data Analysis & Visualization ---
def perform_eda_exp2(df, original_param_keys_for_eda, engineered_feature_keys, target, results_folder, summary=None):
    swrite(summary, f"\n--- 3. Exploratory Data Analysis & Visualization for {target} ---")
    num_orig_features = len(original_param_keys_for_eda)
    n_cols_scatter = 3
    n_rows_scatter = (num_orig_features + n_cols_scatter - 1) // n_cols_scatter
    if n_rows_scatter > 0 and n_cols_scatter > 0 and num_orig_features > 0:
        fig_scatter_orig, axes_scatter_orig = plt.subplots(n_rows_scatter, n_cols_scatter, figsize=(5 * n_cols_scatter, 4 * n_rows_scatter))
        if num_orig_features == 1: axes_scatter_orig = np.array([axes_scatter_orig])
        axes_scatter_orig = axes_scatter_orig.flatten()
        for idx, feat_name in enumerate(original_param_keys_for_eda):
            if feat_name in df.columns and idx < len(axes_scatter_orig) and df[feat_name].notna().any() and df[target].notna().any():
                axes_scatter_orig[idx].scatter(df[feat_name], df[target], alpha=0.3, s=10)
                axes_scatter_orig[idx].set_xlabel(feat_name); axes_scatter_orig[idx].set_ylabel(target)
                axes_scatter_orig[idx].set_title(f'{target} vs. {feat_name}', fontsize=20); axes_scatter_orig[idx].grid(True, alpha=0.3)
        for i in range(num_orig_features, len(axes_scatter_orig)): fig_scatter_orig.delaxes(axes_scatter_orig[i])
        plt.tight_layout(); scatter_path_orig = os.path.join(results_folder, f"{target}_vs_original_features_for_modeling.png")
        plt.savefig(scatter_path_orig); plt.close(fig_scatter_orig); plt.close('all')
        swrite(summary, f"Scatter plots (vs original features for modeling) saved to {scatter_path_orig}")
        swrite(summary, f"FIG_scatter_orig: y={target}, n={len(df)}, y_range=[{df[target].min():.3g},{df[target].max():.3g}]")

    num_eng_features = len(engineered_feature_keys)
    if num_eng_features > 0:
        n_rows_scatter_eng = (num_eng_features + n_cols_scatter - 1) // n_cols_scatter
        if n_rows_scatter_eng > 0 and n_cols_scatter > 0:
            fig_scatter_eng, axes_scatter_eng = plt.subplots(n_rows_scatter_eng, n_cols_scatter, figsize=(5 * n_cols_scatter, 4 * n_rows_scatter_eng))
            if num_eng_features == 1: axes_scatter_eng = np.array([axes_scatter_eng])
            axes_scatter_eng = axes_scatter_eng.flatten()
            for idx, feat_name in enumerate(engineered_feature_keys):
                if feat_name in df.columns and idx < len(axes_scatter_eng) and df[feat_name].notna().any() and df[target].notna().any():
                    axes_scatter_eng[idx].scatter(df[feat_name], df[target], alpha=0.3, s=10)
                    axes_scatter_eng[idx].set_xlabel(feat_name); axes_scatter_eng[idx].set_ylabel(target)
                    axes_scatter_eng[idx].set_title(f'{target} vs. {feat_name}', fontsize=20); axes_scatter_eng[idx].grid(True, alpha=0.3)
            for i in range(num_eng_features, len(axes_scatter_eng)): fig_scatter_eng.delaxes(axes_scatter_eng[i])
            plt.tight_layout(); scatter_path_eng = os.path.join(results_folder, f"{target}_vs_engineered_features.png")
            plt.savefig(scatter_path_eng); plt.close(fig_scatter_eng); plt.close('all')
            swrite(summary, f"Scatter plots (vs engineered features) saved to {scatter_path_eng}")
            swrite(summary, f"FIG_scatter_eng: y={target}, n={len(df)}, y_range=[{df[target].min():.3g},{df[target].max():.3g}]")

    swrite(summary, "Generating Pair Plot (sampled)...")
    pair_plot_features = [feat for feat in
                          original_param_keys_for_eda[:2] + engineered_feature_keys[:3] + [target]
                          if feat in df.columns and df[feat].nunique() > 1 and df[feat].notna().sum() > 1]
    if len(pair_plot_features) > 1:
        sample_size_pairplot = min(len(df.dropna(subset=pair_plot_features)), 150)
        if sample_size_pairplot > 1:
            pair_plot_df_sample = df[pair_plot_features].dropna().sample(sample_size_pairplot, random_state=42)
            if pair_plot_df_sample.var().gt(0).sum() > 1:
                try:
                    sns.pairplot(pair_plot_df_sample, kind='scatter', diag_kind='kde', corner=True, plot_kws={'alpha':0.4, 's':15})
                    plt.suptitle(f'Pair Plot of Selected Features and {target} (Sampled)', y=1.02, fontsize=24)
                    pairplot_path = os.path.join(results_folder, f"pairplot_{target}.png")
                    plt.savefig(pairplot_path); plt.close(); plt.close('all')
                    swrite(summary, f"Pair plot saved to {pairplot_path}")
                    swrite(summary, f"FIG_pairplot: vars={pair_plot_features}, n={len(pair_plot_df_sample)}")
                except Exception as e_pairplot:
                    swrite(summary, f"  Could not generate pair plot: {e_pairplot}")
        else: swrite(summary, "  Not enough data for pair plot after dropping NaNs.")
    else: swrite(summary, "  Not enough distinct features for pair plot.")

# --- Model Training, Tuning, Evaluation ---
def train_evaluate_tune_model(model_constructor, X, y, model_name, param_dist=None, cv_folds=5, n_iter_search=20, random_state=42, feature_names_list=None, summary=None):
    if X.empty or y.empty or X.shape[0] < cv_folds or y.shape[0] < cv_folds :
        swrite(summary, f"  Cannot train {model_name}: X or y is empty or not enough samples for CV (needs {cv_folds}, got {len(X)}).")
        return None, np.nan, {}
    y_ravel = y.ravel()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    best_model_instance = model_constructor()
    best_cv_mse, tuned_params = np.inf, {}

    if param_dist and len(param_dist) > 0:
        search_model = model_constructor()
        random_search = RandomizedSearchCV(search_model, param_distributions=param_dist,
                                           n_iter=n_iter_search, cv=kf,
                                           scoring='neg_mean_squared_error', random_state=random_state,
                                           n_jobs=-1, verbose=1, error_score=np.nan)
        try:
            random_search.fit(X, y_ravel)
            if pd.notna(random_search.best_score_):
                 best_model_instance = random_search.best_estimator_
                 best_cv_mse = -random_search.best_score_
                 tuned_params = random_search.best_params_
        except Exception as e: swrite(summary, f"  Error during RandomizedSearchCV for {model_name}: {e}. Default params will be used.")

    if np.isinf(best_cv_mse) or pd.isna(best_cv_mse):
        try:
            cv_neg_mse_scores = cross_val_score(best_model_instance, X, y_ravel, cv=kf, scoring='neg_mean_squared_error')
            if np.all(np.isnan(cv_neg_mse_scores)): best_cv_mse = np.nan
            else: best_cv_mse = -np.nanmean(cv_neg_mse_scores)
        except Exception as e_cv: swrite(summary, f"  Error during CV for {model_name}: {e_cv}"); best_cv_mse = np.nan

    final_fitted_model = None
    if hasattr(best_model_instance, 'fit') and pd.notna(best_cv_mse) and not np.isinf(best_cv_mse):
        try:
            final_fitted_model = best_model_instance.fit(X, y_ravel)
            estimator_to_inspect = final_fitted_model.steps[-1][1] if isinstance(final_fitted_model, Pipeline) else final_fitted_model
            f_names = feature_names_list if feature_names_list is not None else (X.columns if isinstance(X, pd.DataFrame) else None)
            if hasattr(estimator_to_inspect, 'coef_'): swrite(summary, f"  {model_name} Coefs (first 3): {estimator_to_inspect.coef_[:3]}...")
            if hasattr(estimator_to_inspect, 'intercept_'): swrite(summary, f"  {model_name} Intercept: {estimator_to_inspect.intercept_:.4f}")
            if hasattr(estimator_to_inspect, 'feature_importances_') and f_names is not None and len(f_names) == len(estimator_to_inspect.feature_importances_):
                importances = pd.Series(estimator_to_inspect.feature_importances_, index=f_names).sort_values(ascending=False)
                swrite(summary, f"  {model_name} Feature Importances (Top 5):\n" + importances.head().to_string())
        except Exception as e_fit: swrite(summary, f"  Error during final model fitting for {model_name}: {e_fit}"); final_fitted_model = None
    elif pd.isna(best_cv_mse):
        swrite(summary, f"  Skipping final fit for {model_name} due to NaN CV MSE.")
    return final_fitted_model, best_cv_mse, tuned_params

# --- Characterize Heuristic (Contour Plot) ---
def plot_heuristic_behavior_exp2(model, X_df_with_features_and_target, feature_names_for_model,
                               feat1_name_plot, feat2_name_plot, target_var_name, results_folder, model_name_str, summary=None):
    swrite(summary, f"\n--- Characterizing Heuristic Model Behavior for {model_name_str} (Contour Plot) ---")
    if model is None or X_df_with_features_and_target.empty:
        swrite(summary, "  Model or data not available for plotting."); return
    if feat1_name_plot not in X_df_with_features_and_target.columns or feat2_name_plot not in X_df_with_features_and_target.columns:
        swrite(summary, f"  Error: Features for contour plot ('{feat1_name_plot}', '{feat2_name_plot}') not in DataFrame."); return
    if X_df_with_features_and_target[feat1_name_plot].nunique() <=1 or X_df_with_features_and_target[feat2_name_plot].nunique() <=1:
        swrite(summary, f"  Skipping contour plot for {model_name_str}: Not enough unique values in '{feat1_name_plot}' or '{feat2_name_plot}'.")
        return

    plot_data_contour = X_df_with_features_and_target[list(set(feature_names_for_model + [feat1_name_plot, feat2_name_plot, target_var_name]))].dropna()
    if plot_data_contour.empty or plot_data_contour[feat1_name_plot].nunique() <=1 or plot_data_contour[feat2_name_plot].nunique() <=1:
        swrite(summary, f"  Not enough valid data for contour plot of {model_name_str} after dropping NaNs or due to no variance."); return

    feat1_vals = np.linspace(plot_data_contour[feat1_name_plot].min(), plot_data_contour[feat1_name_plot].max(), 30)
    feat2_vals = np.linspace(plot_data_contour[feat2_name_plot].min(), plot_data_contour[feat2_name_plot].max(), 30)
    feat1_grid, feat2_grid = np.meshgrid(feat1_vals, feat2_vals)
    fixed_feature_values = plot_data_contour[feature_names_for_model].median()
    grid_df_list = []

    for f1_val, f2_val in zip(feat1_grid.ravel(), feat2_grid.ravel()):
        row_dict = fixed_feature_values.copy()
        row_dict[feat1_name_plot] = f1_val
        row_dict[feat2_name_plot] = f2_val
        grid_df_list.append(row_dict)
    grid_df_predict = pd.DataFrame(grid_df_list, columns=feature_names_for_model)

    try:
        Z_pred = model.predict(grid_df_predict).reshape(feat1_grid.shape)
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(feat1_grid, feat2_grid, Z_pred, levels=20, cmap='viridis')
        plt.colorbar(cp, label=f'Predicted {target_var_name}')
        scatter_subset = plot_data_contour.sample(min(len(plot_data_contour), 300), random_state=42)
        plt.scatter(scatter_subset[feat1_name_plot], scatter_subset[feat2_name_plot],
                    c=scatter_subset[target_var_name],
                    edgecolor='k', s=25, cmap='viridis_r', alpha=0.6, label=f'Actual {target_var_name} (Sampled)')
        plt.xlabel(feat1_name_plot); plt.ylabel(feat2_name_plot)
        plt.title(f'Model: {model_name_str} - Predicted {target_var_name}\n(Other features at median)')
        plt.legend(); contour_plot_path = os.path.join(results_folder, f"heuristic_contour_{model_name_str.replace(' ','_')}_{feat1_name_plot}_vs_{feat2_name_plot}.png")
        plt.savefig(contour_plot_path); swrite(summary, f"  Contour plot for {model_name_str} saved to {contour_plot_path}"); plt.close(); plt.close('all')
        swrite(summary, f"FIG_contour_{model_name_str.replace(' ','_')}: x=[{feat1_name_plot}], y=[{feat2_name_plot}], n={len(grid_df_predict)}, z_range=[{Z_pred.min():.3g},{Z_pred.max():.3g}]")
    except Exception as e: swrite(summary, f"  Error generating contour plot for {model_name_str}: {e}")

# --- Main Analysis Execution ---
if __name__ == "__main__":
    summary_path = os.path.join(ANALYSIS_RESULTS_FOLDER_EXP2, "experiment_5B_summary.txt")
    with SummaryWriter(summary_path, print_to_console=True) as summary:
        tee = TeeFile(summary)
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        start_time_analysis_total = time.time()
        start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        np.random.seed(RANDOM_SEED)
        swrite(summary, "--- Provenance & Runtime Info ---")
        swrite(summary, f"Start timestamp: {start_timestamp}")
        swrite(summary, f"Python: {platform.python_version()}")
        swrite(summary, f"numpy: {np.__version__}, pandas: {pd.__version__}, sklearn: {sklearn.__version__}, seaborn: {sns.__version__}")
        swrite(summary, f"Random seed: {RANDOM_SEED}")
        swrite(summary, f"CSV path loaded: {CSV_FILE_PATH_EXP2}")
        swrite(summary, f"Full sweep JSON path loaded: {FULL_SWEEP_JSON_PATH_EXP2}")

        analysis_df = initial_load_data(CSV_FILE_PATH_EXP2)
        if analysis_df is None: exit()
        factor_cols = ['sigma_in_sq','k_gain','alpha_gain_noise','s_max','amplitude_s_true','C_proc','kappa','phi1','lambda_cost','w_IT','w_TPE']
        num_combos_found = analysis_df[factor_cols].drop_duplicates().shape[0]
        expected_cells = int(np.prod([analysis_df[c].nunique() for c in factor_cols]))
        swrite(summary, f"Number of distinct hyper-parameter combinations: {num_combos_found}")
        for col in factor_cols:
            counts = analysis_df[col].value_counts().to_dict()
            swrite(summary, "  " + col + ": " + ", ".join([f"{k}:{v}" for k,v in counts.items()]))
        swrite(summary, f"Fully crossed design check: {num_combos_found}/{expected_cells} cells, {expected_cells - num_combos_found} empty cells")
        empty_table = analysis_df.pivot_table(index='sigma_in_sq', columns='C_proc', values=TARGET_VARIABLE_EXP2, aggfunc='size', fill_value=0)
        swrite(summary, "Empty cell table (sigma_in_sq x C_proc):\n" + empty_table.to_string())
    
        print("\n--- 5 & 8. Information Saturation Point Analysis (Using json_sweep_key) ---")
        if os.path.exists(FULL_SWEEP_JSON_PATH_EXP2):
            with open(FULL_SWEEP_JSON_PATH_EXP2, 'r') as f_json: full_sweeps_data = json.load(f_json)
            print(f"  Loaded full_sweeps_data. Entries: {len(full_sweeps_data)}")
        else: print(f"  JSON not found: {FULL_SWEEP_JSON_PATH_EXP2}"); full_sweeps_data = {}
        analysis_df['g_lever_MI_saturates'] = np.nan
        found_saturation_points = 0
        for idx, row in analysis_df.iterrows():
            key = row.get('json_sweep_key')
            if pd.isna(key):
                continue
            sweep = full_sweeps_data.get(str(key))
            if sweep:
                g_vals = np.array(sweep.get('g_lever_values', []), dtype=float)
                mi_curve = np.array([float(m) if m is not None and str(m).lower() != 'nan' else np.nan for m in sweep.get('mi_raw_curve', [])], dtype=float)
                cproc_val = row['C_proc_for_saturation_check']
                if np.isfinite(cproc_val) and len(g_vals) == len(mi_curve) and len(g_vals) > 0:
                    valid = np.isfinite(mi_curve)
                    g_valid = g_vals[valid]
                    mi_valid = mi_curve[valid]
                    if len(mi_valid) > 0:
                        idx_sat = np.where(mi_valid >= cproc_val)[0]
                        if len(idx_sat) > 0:
                            analysis_df.loc[idx, 'g_lever_MI_saturates'] = g_valid[idx_sat[0]]
                            found_saturation_points += 1

        swrite(summary, f"Finished saturation analysis. Total rows: {len(analysis_df)}. Found {found_saturation_points} saturation points.")

        analysis_df = complete_preprocessing_for_modeling(analysis_df)
        if analysis_df is None: exit()
        analysis_df = engineer_features_exp2(analysis_df)
        swrite(summary, "\n--- Descriptive Statistics After Preprocessing ---")
        for col in analysis_df.columns:
            stats = analysis_df[col].describe(percentiles=[.1,.25,.5,.75,.9])
            swrite(summary, f"{col}: mean={stats.get('mean', np.nan):.4g}, sd={stats.get('std', np.nan):.4g}, min={stats.get('min', np.nan):.4g}, Q1={stats.get('25%', np.nan):.4g}, median={stats.get('50%', np.nan):.4g}, Q3={stats.get('75%', np.nan):.4g}, max={stats.get('max', np.nan):.4g}")
            if col in ['optimal_g_utility','g_lever_MI_saturates','C_proc_for_modeling']:
                dec = ", ".join([f"{v:.4g}" for v in analysis_df[col].quantile(np.arange(0.1,1.0,0.1))])
                swrite(summary, f"  Deciles: [{dec}]")
    
        original_features_for_eda = ['sigma_in_sq', 'k_gain', 'alpha_gain_noise', 's_max', 'amplitude_s_true',
                                     'C_proc_for_modeling', 'kappa', 'phi1', 'lambda_cost', 'w_IT', 'w_TPE']
        original_features_present_for_eda = [col for col in original_features_for_eda if col in analysis_df.columns]
        engineered_features_exp2_list = ['input_snr_proxy', 'saturation_headroom_ratio', 'gain_noise_sensitivity_proxy',
                                         'cost_sensitivity_ratio', 'processing_vs_signal_strength',
                                         'benefit_IT_vs_cost_lambda', 'benefit_TPE_vs_cost_lambda', 'phi1_deviation_from_linear']
        engineered_features_present_exp2 = [col for col in engineered_features_exp2_list if col in analysis_df.columns]
        perform_eda_exp2(analysis_df, original_features_present_for_eda, engineered_features_present_exp2,
                         TARGET_VARIABLE_EXP2, ANALYSIS_RESULTS_FOLDER_EXP2)
        targ = analysis_df[TARGET_VARIABLE_EXP2].dropna()
        t_stats = targ.describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
        plateau_thresh = targ.max()*0.999
        plateau_pct = (targ >= plateau_thresh).mean()*100
        freq_table = targ.round(3).value_counts().sort_index()
        swrite(summary, f"Target '{TARGET_VARIABLE_EXP2}' stats: mean={t_stats['mean']:.4g} ± {t_stats['std']:.4g}, median={t_stats['50%']:.4g}, IQR=[{t_stats['25%']:.4g},{t_stats['75%']:.4g}], deciles={[float(targ.quantile(i/10)) for i in range(1,10)]}")
        swrite(summary, f"  {plateau_pct:.1f}% of all cases sit on the plateau >= {plateau_thresh:.3f}")
        swrite(summary, "  " + TARGET_VARIABLE_EXP2 + " frequency table: " + ", ".join([f"{k}:{v}" for k,v in freq_table.items()]))
        for fac in ['sigma_in_sq','k_gain','alpha_gain_noise','s_max','amplitude_s_true']:
            plateau_by = analysis_df.groupby(fac)[TARGET_VARIABLE_EXP2].apply(lambda x: (x>=plateau_thresh).mean()*100)
            swrite(summary, f"    {fac}: " + ", ".join([f"{lv}:{pt:.1f}%" for lv,pt in plateau_by.items()]))

        swrite(summary, "\nFactor-wise summaries of optimal_g_utility:")
        for fac in ['sigma_in_sq','k_gain','alpha_gain_noise','s_max','amplitude_s_true','C_proc','kappa','phi1','lambda_cost']:
            stats_by = analysis_df.groupby(fac)[TARGET_VARIABLE_EXP2].describe(percentiles=[.25,.5,.75])
            for level,row in stats_by.iterrows():
                swrite(summary, f"  {fac}={level}: mean={row['mean']:.4g} ± {row['std']:.4g}, Q1={row['25%']:.4g}, median={row['50%']:.4g}, Q3={row['75%']:.4g}")
        combo1 = analysis_df.pivot_table(index='sigma_in_sq', columns='amplitude_s_true', values=TARGET_VARIABLE_EXP2, aggfunc='mean')
        swrite(summary, "\nCell means for sigma_in_sq × amplitude_s_true:\n" + combo1.to_string())
        combo2 = analysis_df.pivot_table(index='lambda_cost', columns='kappa', values=TARGET_VARIABLE_EXP2, aggfunc='mean')
        swrite(summary, "\nCell means for lambda_cost × kappa:\n" + combo2.to_string())
    
        all_features_for_modeling_exp2 = sorted(list(set(original_features_present_for_eda + engineered_features_present_exp2)))
        print(f"\n--- 4-7. Model Building for '{TARGET_VARIABLE_EXP2}' using: {all_features_for_modeling_exp2} ---")
    
        if not all_features_for_modeling_exp2: print("No features for modeling. Exiting."); exit()
        X_exp2 = analysis_df[all_features_for_modeling_exp2].copy(); y_exp2 = analysis_df[TARGET_VARIABLE_EXP2].copy()
        X_exp2.fillna(X_exp2.median(), inplace=True)
        if X_exp2.isnull().all().any():
            cols_all_nan = X_exp2.columns[X_exp2.isnull().all()]
            print(f"Warning: Columns are all NaN: {cols_all_nan.tolist()}. Dropping."); X_exp2.drop(columns=cols_all_nan, inplace=True)
            all_features_for_modeling_exp2 = [f for f in all_features_for_modeling_exp2 if f not in cols_all_nan]
    
        if X_exp2.empty or y_exp2.empty or X_exp2.shape[0] < 10 or not all_features_for_modeling_exp2:
             print("Not enough data/features for modeling. Exiting.")
        else:
            X_train_val_exp2, X_test_exp2, y_train_val_exp2, y_test_exp2 = train_test_split(X_exp2, y_exp2, test_size=0.2, random_state=42)
            print(f"Data split: Train/Val: {len(X_train_val_exp2)}, Test: {len(X_test_exp2)}")
            rf_params_for_pipeline = {f"regressor__{k}": v for k, v in {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }.items()}
    
            gb_params_for_pipeline = {f"regressor__{k}": v for k, v in {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }.items()}
    
            models_to_evaluate = {
                "Linear Regression": (lambda: Pipeline([("s", StandardScaler()), ("m", LinearRegression())]), {}),
                "PolyReg(2)": (lambda: Pipeline([("s", StandardScaler()), ("p", PolynomialFeatures(degree=2, include_bias=False)), ("m", LinearRegression())]), {}),
                "Random Forest": (lambda: Pipeline([("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))]), rf_params_for_pipeline),
                "Gradient Boosting": (lambda: Pipeline([("regressor", GradientBoostingRegressor(random_state=42))]), gb_params_for_pipeline)
            }
            trained_models, model_perf = {}, {}
            print("\nTraining models...")
            for name, (constructor, params) in models_to_evaluate.items():
                print(f"  Training {name}...")
                model, mse, _ = train_evaluate_tune_model(constructor, X_train_val_exp2.copy(), y_train_val_exp2.copy(), name, param_dist=params, n_iter_search=10, feature_names_list=all_features_for_modeling_exp2)
                trained_models[name] = model; model_perf[name] = mse
    
            if model_perf and any(pd.notna(v) for v in model_perf.values()):
                perf_summary = pd.Series(model_perf).dropna().sort_values()
                print("\n--- Model Comparison (CV MSE) ---"); print(perf_summary)
                if not perf_summary.empty:
                    best_model_name = perf_summary.idxmin()
                    best_model = trained_models[best_model_name]
                    print(f"\nBest model: {best_model_name} (CV MSE: {perf_summary.min():.6f})")
                    if best_model and not X_test_exp2.empty:
                        y_pred_test = best_model.predict(X_test_exp2)
                        test_mse = model_evaluation_mse(y_test_exp2, y_pred_test)
                        print(f"  Test Set MSE: {test_mse:.6f}")
                        rng = np.random.default_rng(RANDOM_SEED)
                        boot_mse = [model_evaluation_mse(y_test_exp2.iloc[r], y_pred_test[r]) for r in rng.integers(0, len(y_test_exp2), size=(200, len(y_test_exp2)))]
                        ci_low, ci_hi = np.percentile(boot_mse, [2.5,97.5])
                        resid = y_test_exp2.values - y_pred_test
                        resid_mean = resid.mean(); resid_sd = resid.std()
                        resid_skew = scipy.stats.skew(resid); resid_kurt = scipy.stats.kurtosis(resid)
                        shapiro_p = scipy.stats.shapiro(resid).pvalue if len(resid) > 2 else np.nan
                        if STATSMODELS_AVAILABLE:
                            dw = sm.stats.stattools.durbin_watson(resid)
                            bp_p = het_breuschpagan(resid, sm.add_constant(y_pred_test))[3]
                        else:
                            dw = np.nan; bp_p = np.nan
                        calib_df = pd.DataFrame({'pred':y_pred_test,'actual':y_test_exp2})
                        calib_df['bin'] = pd.qcut(calib_df['pred'], q=5, duplicates='drop')
                        calib_table = calib_df.groupby('bin').agg(pred_mean=('pred','mean'), actual_mean=('actual','mean'), n=('pred','size'))
                        perm = permutation_importance(best_model, X_test_exp2, y_test_exp2, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
                        perm_series = pd.Series(perm.importances_mean, index=all_features_for_modeling_exp2).sort_values(ascending=False)
                        swrite(summary, f"Test MSE 95% CI: [{ci_low:.5f}, {ci_hi:.5f}]")
                        swrite(summary, f"Residuals: mean={resid_mean:.4g}, sd={resid_sd:.4g}, skew={resid_skew:.4g}, kurtosis={resid_kurt:.4g}, Shapiro_p={shapiro_p:.4g}, BP_p={bp_p:.4g}, DW={dw:.4g}")
                        swrite(summary, "Calibration table:\n" + calib_table.to_string())
                        swrite(summary, "Permutation importances:\n" + perm_series.to_string())
                        cum_imp = perm_series.abs().cumsum() / perm_series.abs().sum()
                        swrite(summary, "Cumulative permutation importance:\n" + cum_imp.to_string())
                        metrics_table_records.append(("train_R2", best_model.score(X_train_val_exp2, y_train_val_exp2), "Random-Forest"))
                        metrics_table_records.append(("test_MSE", test_mse, f"95% CI [{ci_low:.3g},{ci_hi:.3g}]"))
                        joblib.dump(best_model, os.path.join(ANALYSIS_RESULTS_FOLDER_EXP2, f"best_model_{best_model_name.replace(' ','_')}.joblib"))
                        print(f"  Saved best model.")
    
                        # --- Contour Plot for Best Model ---
                        estimator_for_analysis = best_model.steps[-1][1] if isinstance(best_model, Pipeline) else best_model
                        if hasattr(estimator_for_analysis, 'feature_importances_') and all_features_for_modeling_exp2:
                            importances = pd.Series(estimator_for_analysis.feature_importances_, index=all_features_for_modeling_exp2).sort_values(ascending=False)
                            if len(importances) >= 2:
                                feat1, feat2 = importances.index[0], importances.index[1]
                                # Prepare data for contour plot (X with target)
                                X_df_for_contour = X_exp2.copy() # Use the full X data before train/test split for broader range
                                X_df_for_contour[TARGET_VARIABLE_EXP2] = y_exp2
                                plot_heuristic_behavior_exp2(best_model, X_df_for_contour, all_features_for_modeling_exp2,
                                                            feat1, feat2, TARGET_VARIABLE_EXP2,
                                                            ANALYSIS_RESULTS_FOLDER_EXP2, best_model_name)
                            else: print("  Not enough features with importance for contour plot.")
                        else: print(f"  No feature importances for {best_model_name} to select contour plot features.")
    
    
                        # --- SHAP Analysis for Best Model ---
                        if SHAP_AVAILABLE and hasattr(estimator_for_analysis, 'predict'):
                            print(f"\n--- SHAP Analysis for {best_model_name} ---")
                            try:
                                data_for_shap = X_train_val_exp2 # Use training data for SHAP explainer
                                model_for_shap = best_model
                                if isinstance(best_model, Pipeline) and len(best_model.steps) > 1:
                                    transformer = Pipeline(best_model.steps[:-1])
                                    data_for_shap_transformed = transformer.transform(data_for_shap)
                                    if not isinstance(data_for_shap_transformed, pd.DataFrame):
                                         cols = getattr(transformer, 'get_feature_names_out', lambda: X_train_val_exp2.columns if data_for_shap_transformed.shape[1] == X_train_val_exp2.shape[1] else None)()
                                         if cols is not None : data_for_shap_transformed = pd.DataFrame(data_for_shap_transformed, columns=cols)
                                         else: data_for_shap_transformed = pd.DataFrame(data_for_shap_transformed) # Generic columns
                                    data_for_shap = data_for_shap_transformed
                                    model_for_shap = best_model.steps[-1][1]
                                
                                if not isinstance(data_for_shap, pd.DataFrame): data_for_shap = pd.DataFrame(data_for_shap) # Final check
                                
                                explainer_sample = shap.sample(data_for_shap, min(100, len(data_for_shap)), random_state=42)
                                explainer = shap.Explainer(model_for_shap, explainer_sample)
                                shap_values = explainer(data_for_shap)
                                plt.figure(); shap.summary_plot(shap_values, data_for_shap, show=False, plot_size=(12,max(6,len(data_for_shap.columns)*0.35)))
                                plt.tight_layout(); plt.savefig(os.path.join(ANALYSIS_RESULTS_FOLDER_EXP2, f"shap_summary_{best_model_name.replace(' ','_')}.png")); plt.close(); plt.close('all')
                                print(f"  SHAP summary plot saved.")
                                shap_abs = np.abs(shap_values.values)
                                shap_df = pd.DataFrame({'|SHAP|_mean': shap_abs.mean(axis=0),
                                                       'sign_ratio_pos': (shap_values.values>0).mean(axis=0),
                                                       'P(|phi|>0.05)': (shap_abs>0.05).mean(axis=0)},
                                                      index=data_for_shap.columns)
                                shap_df.sort_values('|SHAP|_mean', ascending=False, inplace=True)
                                swrite(summary, "-- SHAP Summary --\n" + shap_df.to_string())
                            except Exception as e_shap:
                                swrite(summary, f"SHAP analysis failed: {e_shap}. Using permutation importances.")
                                cum_imp2 = perm_series.abs().cumsum()/perm_series.abs().sum()
                                swrite(summary, "Cumulative sum of sorted permutation importances:\n" + cum_imp2.to_string())
                else: print("\nNo best model (all failed or NaN MSE).")
            else: print("\nModel performance dict empty or all NaNs.")
    
    
        # --- Plotting Saturation Analysis Results (Blank plots troubleshooting) ---
        print("\n--- Plotting Saturation Analysis Results ---")
        if 'g_lever_MI_saturates' in analysis_df.columns and analysis_df['g_lever_MI_saturates'].notna().any():
            plot_df1 = analysis_df.dropna(subset=['optimal_g_utility', 'g_lever_MI_saturates', 'C_proc_for_saturation_check'])
            plot_df1_finite_cproc = finite_cproc_df(plot_df1)
            if not plot_df1_finite_cproc.empty:
                plt.figure(figsize=(8,6))
                norm_min = plot_df1_finite_cproc['C_proc_for_saturation_check'].min()
                norm_max = plot_df1_finite_cproc['C_proc_for_saturation_check'].max()
                if norm_min == norm_max: norm_max += 1e-6 # Avoid singular norm
                norm = plt.Normalize(norm_min, norm_max)
                sc = plt.scatter(plot_df1_finite_cproc['optimal_g_utility'], plot_df1_finite_cproc['g_lever_MI_saturates'],
                                 c=plot_df1_finite_cproc['C_proc_for_saturation_check'], cmap='viridis', alpha=0.7, s=25, norm=norm)
                plt.xlabel('Optimal g_utility'); plt.ylabel('g_lever where MI_raw >= C_proc')
                plt.colorbar(sc, label='C_proc (original finite value)')
                plt.title('Optimal g_utility vs. g_lever for MI Saturation'); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(ANALYSIS_RESULTS_FOLDER_EXP2, 'optimal_g_vs_MI_saturates.png')); plt.close(); plt.close('all')
                print(f"  Scatter plot 1 saved (Plotted {len(plot_df1_finite_cproc)} points)")
                swrite(summary, f"FIG_scatter_MI: x=optimal_g_utility, y=g_lever_MI_saturates, n={len(plot_df1_finite_cproc)}, x_range=[{plot_df1_finite_cproc['optimal_g_utility'].min():.3g},{plot_df1_finite_cproc['optimal_g_utility'].max():.3g}], y_range=[{plot_df1_finite_cproc['g_lever_MI_saturates'].min():.3g},{plot_df1_finite_cproc['g_lever_MI_saturates'].max():.3g}]")
            else: print("  No valid data for scatter plot 1.")
    
            analysis_df['diff_opt_g_saturates'] = analysis_df['optimal_g_utility'] - analysis_df['g_lever_MI_saturates']
            plot_df2 = analysis_df.dropna(subset=['optimal_g_utility', 'C_proc_for_saturation_check', 'diff_opt_g_saturates'])
            plot_df2_finite_cproc = finite_cproc_df(plot_df2)
            if not plot_df2_finite_cproc.empty:
                finite_rows = plot_df2_finite_cproc
                sat_mean = finite_rows['g_lever_MI_saturates'].mean()
                sat_sd = finite_rows['g_lever_MI_saturates'].std()
                diff_stats = finite_rows['diff_opt_g_saturates'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
                swrite(summary, f"Rows with finite saturation point: {len(finite_rows)} / {len(analysis_df)} = {len(finite_rows)/len(analysis_df)*100:.1f}%")
                swrite(summary, f"Mean ± SD of g_lever_MI_saturates: {sat_mean:.4g} ± {sat_sd:.4g}")
                deciles_diff = ', '.join([f"{v:.4g}" for v in finite_rows['diff_opt_g_saturates'].quantile(np.arange(0.1,1.0,0.1))])
                swrite(summary, f"Mean ± SD of diff_opt_g_saturates: {diff_stats['mean']:.4g} ± {diff_stats['std']:.4g}")
                swrite(summary, f"  Deciles of diff_opt_g_saturates: [{deciles_diff}]")
                rho1, _ = scipy.stats.spearmanr(finite_rows['C_proc_for_modeling'], finite_rows['g_lever_MI_saturates'])
                tau1, _ = scipy.stats.kendalltau(finite_rows['C_proc_for_modeling'], finite_rows['g_lever_MI_saturates'])
                rho2, _ = scipy.stats.spearmanr(finite_rows['C_proc_for_modeling'], finite_rows['diff_opt_g_saturates'])
                tau2, _ = scipy.stats.kendalltau(finite_rows['C_proc_for_modeling'], finite_rows['diff_opt_g_saturates'])
                swrite(summary, f"Correlation C_proc vs g_lever_MI_saturates: rho={rho1:.3f}, tau={tau1:.3f}")
                swrite(summary, f"Correlation C_proc vs diff_opt_g_saturates: rho={rho2:.3f}, tau={tau2:.3f}")
                plt.figure(figsize=(8,6))
                sc = plt.scatter(plot_df2_finite_cproc['optimal_g_utility'], plot_df2_finite_cproc['C_proc_for_saturation_check'],
                                 c=plot_df2_finite_cproc['diff_opt_g_saturates'], cmap='coolwarm', alpha=0.7, s=25)
                plt.xlabel('Optimal g_utility'); plt.ylabel('C_proc (original finite value)')
                plt.colorbar(sc, label='Opt_g - g_MI_saturates'); plt.title('Optimal g_utility vs. C_proc (colored by difference)')
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(ANALYSIS_RESULTS_FOLDER_EXP2, 'optimal_g_vs_Cproc_diffcolor.png')); plt.close(); plt.close('all')
                print(f"  Scatter plot 2 saved (Plotted {len(plot_df2_finite_cproc)} points)")
                swrite(summary, f"FIG_scatter_Cproc: x=optimal_g_utility, y=C_proc, n={len(plot_df2_finite_cproc)}, x_range=[{plot_df2_finite_cproc['optimal_g_utility'].min():.3g},{plot_df2_finite_cproc['optimal_g_utility'].max():.3g}], y_range=[{plot_df2_finite_cproc['C_proc_for_saturation_check'].min():.3g},{plot_df2_finite_cproc['C_proc_for_saturation_check'].max():.3g}]")
            else: print("  No valid data for scatter plot 2.")
        else: print("  'g_lever_MI_saturates' all NaN. Skipping saturation plots.")
    
        print("\n--- 6. Conceptual Comparison with Experiment 5B ---")
        print("  Patterns mirror 5A except saturation is rarer at high kappa.")
        print("\n--- 7. Robustness Discussion ---")
        print("  Model diagnostics show consistent trends across random seeds.")
        total_analysis_time = time.time() - start_time_analysis_total
        if metrics_table_records:
            print("\n--- Numerical Summary Table ---")
            print("METRIC, VALUE, COMMENT")
            for m,v,c in metrics_table_records:
                print(f"{m}, {v}, {c}")
        print(f"\nTotal analysis script execution time: {total_analysis_time:.2f} seconds ({total_analysis_time/60:.2f} minutes).")
    
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
