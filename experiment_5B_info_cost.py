# experiment_2_info_cost.py
import numpy as np
import matplotlib.pyplot as plt

# Double all default font sizes across generated figures
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
import os
import json # For saving MI_raw curves

RNG = np.random.default_rng(123)

# --- Configuration Parameters ---
# Signal parameters
DEFAULT_AMPLITUDE_S_TRUE = 1.0
DEFAULT_FREQUENCY_S_TRUE = 1.0  # Hz
DURATION_T = 2.0  # seconds
SAMPLING_RATE_T = 100  # Hz
NUM_POINTS = int(DURATION_T * SAMPLING_RATE_T)
T_VECTOR = np.linspace(0, DURATION_T, NUM_POINTS, endpoint=False)

# gLever sweep parameters
G_LEVER_VALUES_FOR_SWEEP = np.linspace(0.1, 8.0, 50)
NUM_TRIALS_PER_G_VALUE = 15 # Default: 15

# --- Results Folder ---
RESULTS_FOLDER = "results_experiment_5B_info_cost"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# --- Helper Functions ---
def generate_true_signal(t_vector, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t_vector)

def generate_input_noise(num_points, sigma_in_sq):
    std_dev_in = np.sqrt(max(0, sigma_in_sq))
    return RNG.normal(0, std_dev_in, num_points)

def generate_gain_noise(num_points, g_lever, alpha_gain_noise, k_gain, sigma_in_sq_base):
    if g_lever < 1e-9: var_Ng = 0.0
    else: var_Ng = k_gain * (g_lever**alpha_gain_noise) * sigma_in_sq_base
    std_dev_Ng = np.sqrt(max(0, var_Ng))
    return RNG.normal(0, std_dev_Ng, num_points)

def clip_signal(signal, s_max_clip_value):
    return np.clip(signal, -s_max_clip_value, s_max_clip_value)

def calculate_mse(S_true, S_hat):
    if S_true is None or S_hat is None or len(S_true) != len(S_hat): return np.nan
    return np.mean((S_true - S_hat)**2)

def run_signal_chain_upto_saturation(S_true_segment, g_lever, sigma_in_sq, k_gain, alpha_gain_noise, s_max_saturation):
    num_points_segment = len(S_true_segment)
    N_in = generate_input_noise(num_points_segment, sigma_in_sq)
    S_obs = S_true_segment + N_in
    S_gained = g_lever * S_obs
    N_g = generate_gain_noise(num_points_segment, g_lever, alpha_gain_noise, k_gain, sigma_in_sq)
    S_noisy_gained = S_gained + N_g
    S_saturated = clip_signal(S_noisy_gained, s_max_saturation)
    return S_saturated

def calculate_mutual_information_binned(signal1, signal2, num_bins=32):
    if signal1 is None or signal2 is None or len(signal1) != len(signal2): return 0.0
    if np.any(np.isnan(signal1)) or np.any(np.isnan(signal2)) or \
       np.any(np.isinf(signal1)) or np.any(np.isinf(signal2)): return 0.0
    s1, s2 = np.asarray(signal1).ravel(), np.asarray(signal2).ravel()
    if len(s1) == 0: return 0.0
    min_s1, max_s1, min_s2, max_s2 = s1.min(), s1.max(), s2.min(), s2.max()
    if abs(min_s1 - max_s1) < 1e-9 or abs(min_s2 - max_s2) < 1e-9 : return 0.0 # If either signal is effectively constant

    bins_s1 = np.linspace(min_s1, max_s1, num_bins + 1)
    bins_s2 = np.linspace(min_s2, max_s2, num_bins + 1)
    try:
        joint_hist, _, _ = np.histogram2d(s1, s2, bins=(bins_s1, bins_s2))
    except Exception: return 0.0

    sum_joint_hist = np.sum(joint_hist)
    if sum_joint_hist == 0: return 0.0
    joint_p = joint_hist / sum_joint_hist

    p_s1, p_s2 = np.sum(joint_p, axis=1), np.sum(joint_p, axis=0)
    nz_s1, nz_s2, nz_joint = p_s1 > 1e-12, p_s2 > 1e-12, joint_p > 1e-12
    h_s1 = -np.sum(p_s1[nz_s1] * np.log2(p_s1[nz_s1])) if np.any(nz_s1) else 0.0
    h_s2 = -np.sum(p_s2[nz_s2] * np.log2(p_s2[nz_s2])) if np.any(nz_s2) else 0.0
    h_s1_s2 = -np.sum(joint_p[nz_joint] * np.log2(joint_p[nz_joint])) if np.any(nz_joint) else 0.0
    mi = h_s1 + h_s2 - h_s1_s2
    return max(0, mi if np.isfinite(mi) else 0.0)


def apply_processing_bottleneck(S_saturated, C_proc_bits, S_true_for_MI_calc):
    mi_raw = calculate_mutual_information_binned(S_true_for_MI_calc, S_saturated)

    if np.isinf(C_proc_bits): # If C_proc is infinite, no bottleneck
        it_final = mi_raw
        S_proc = S_saturated
    else: # Finite C_proc
        it_final = min(mi_raw, C_proc_bits)
        # Ensure it_final is non-negative and finite for 2**it_final
        it_final_for_levels = max(0, it_final if np.isfinite(it_final) else 0)
        num_levels_proc = max(2, int(round(2**it_final_for_levels)))

        s_min_sat, s_max_sat = S_saturated.min(), S_saturated.max()

        if abs(s_min_sat - s_max_sat) < 1e-9: # If signal is effectively constant
            S_proc = S_saturated
        else:
            if num_levels_proc == 2:
                threshold = (s_min_sat + s_max_sat) / 2.0
                level_low = s_min_sat
                level_high = s_max_sat if (s_max_sat > s_min_sat + 1e-9) else s_min_sat + 1e-9 # Ensure distinct levels
                S_proc = np.where(S_saturated <= threshold, level_low, level_high)
            elif num_levels_proc > 2 :
                quant_bins = np.linspace(s_min_sat, s_max_sat, num_levels_proc + 1)
                quant_bins = np.unique(quant_bins) # Ensure unique bin edges
                if len(quant_bins) < 2 : # Not enough distinct bins, treat as constant
                    S_proc = np.full_like(S_saturated, np.mean(S_saturated))
                else:
                    # np.digitize: indices are 1-based, so subtract 1 for 0-based array indexing
                    digitized_indices_proc = np.digitize(S_saturated, quant_bins[:-1], right=False) - 1
                    # Clip indices to be within the bounds of bin_centers_proc
                    digitized_indices_proc = np.clip(digitized_indices_proc, 0, len(quant_bins)-2)
                    bin_centers_proc = (quant_bins[:-1] + quant_bins[1:]) / 2.0
                    S_proc = bin_centers_proc[digitized_indices_proc]
            else: # num_levels_proc <= 1 (should be caught by max(2, ...), but as a fallback)
                S_proc = np.full_like(S_saturated, np.mean(S_saturated))

    return S_proc, mi_raw, it_final


def calculate_energetic_cost(g_lever, kappa, phi1):
    return kappa * (g_lever**phi1)

def calculate_utility(it, tpe, eg, w_IT, w_TPE, lambda_cost):
    safe_tpe = max(0, tpe) if not np.isnan(tpe) else 1e6
    it_val = it if not np.isnan(it) else 0
    benefit = (w_IT * it_val) - (w_TPE * safe_tpe)
    return benefit - (lambda_cost * eg)

# --- Experiment 5B Sweep Function ---
def run_exp2_gLever_sweep(
    g_lever_values_sweep, num_trials,
    sigma_in_sq, k_gain, alpha_gain_noise, s_max, C_proc,
    kappa, phi1, lambda_cost,
    w_IT, w_TPE,
    t_vector, amplitude_s_true, frequency_s_true
):
    sweep_results_list = []
    S_true = generate_true_signal(t_vector, amplitude_s_true, frequency_s_true)
    mi_raw_curve_for_this_combo = []

    for g_lever in g_lever_values_sweep:
        trial_tpes, trial_its_final, trial_mi_raws_for_g, trial_utilities = [], [], [], []
        current_eg = calculate_energetic_cost(g_lever, kappa, phi1)

        for _ in range(num_trials):
            S_saturated_trial = run_signal_chain_upto_saturation(
                S_true, g_lever, sigma_in_sq, k_gain, alpha_gain_noise, s_max
            )
            S_proc_trial, mi_raw_trial, it_final_trial = apply_processing_bottleneck(
                S_saturated_trial, C_proc, S_true_for_MI_calc=S_true
            )

            tpe_trial = calculate_mse(S_true, S_proc_trial)
            utility_trial = calculate_utility(it_final_trial, tpe_trial, current_eg, w_IT, w_TPE, lambda_cost)

            trial_tpes.append(tpe_trial)
            trial_its_final.append(it_final_trial)
            trial_mi_raws_for_g.append(mi_raw_trial)
            trial_utilities.append(utility_trial)

        mean_mi_raw_current_g = np.nanmean(trial_mi_raws_for_g) if trial_mi_raws_for_g else np.nan
        mi_raw_curve_for_this_combo.append(mean_mi_raw_current_g)

        sweep_results_list.append({
            'g_lever': g_lever,
            'mean_tpe': np.nanmean(trial_tpes) if trial_tpes else np.nan,
            'mean_it_final': np.nanmean(trial_its_final) if trial_its_final else np.nan,
            'mean_mi_raw': mean_mi_raw_current_g,
            'energetic_cost': current_eg,
            'mean_utility': np.nanmean(trial_utilities) if trial_utilities else np.nan
        })
    df_results = pd.DataFrame(sweep_results_list)
    return df_results, mi_raw_curve_for_this_combo

# --- Main Execution Block for Experiment 5B ---
if __name__ == "__main__":
    param_grid_exp2 = {
        'sigma_in_sq': [0.05**2, 0.15**2], 'k_gain': [0.01, 0.05],
        'alpha_gain_noise': [1.0, 1.5], 's_max': [1.0, 2.5],
        'amplitude_s_true': [0.5, 1.0], 'C_proc': [2.0, 4.0, 6.0, np.inf],
        'kappa': [0.05, 0.1], 'phi1': [1.0, 1.25],
        'lambda_cost': [0.01, 0.05, 0.1], 'w_IT': [1.0], 'w_TPE': [0.5]
    }
    # # Quick test grid
    # param_grid_exp2 = {
    #     'sigma_in_sq': [0.1**2], 'k_gain': [0.01],
    #     'alpha_gain_noise': [1.0],'s_max': [2.5],
    #     'amplitude_s_true': [1.0],'C_proc': [3.0, np.inf], # Test finite and inf C_proc
    #     'kappa': [0.1], 'phi1': [1.0],
    #     'lambda_cost': [0.01],'w_IT': [1.0], 'w_TPE': [0.1]
    # }
    # G_LEVER_VALUES_FOR_SWEEP = np.linspace(0.1, 2.0, 5) # Reduced sweep for testing
    # NUM_TRIALS_PER_G_VALUE = 2 # Reduced trials

    param_keys_exp2, param_value_levels_exp2 = zip(*param_grid_exp2.items())
    parameter_combinations_exp2 = [dict(zip(param_keys_exp2, v)) for v in itertools.product(*param_value_levels_exp2)]

    print(f"Starting Experiment 5B: Information Channel with Energetic Costs & Bottlenecks")
    print(f"Total parameter combinations to test: {len(parameter_combinations_exp2)}")
    print(f"Each combination sweeps {len(G_LEVER_VALUES_FOR_SWEEP)} gLever values, with {NUM_TRIALS_PER_G_VALUE} trials each.")

    all_exp2_optimal_results_list = []
    all_exp2_full_sweep_details = {}
    start_time_total_exp2 = time.time()

    for i, system_params_exp2 in enumerate(parameter_combinations_exp2):
        start_time_combo = time.time()
        progress_pct = (i + 1) / len(parameter_combinations_exp2) * 100
        print(f"Running Exp2 Combo {i+1}/{len(parameter_combinations_exp2)} ({progress_pct:.1f}%) ", end='\r')

        df_sweep_results, mi_raw_curve = run_exp2_gLever_sweep(
            G_LEVER_VALUES_FOR_SWEEP, NUM_TRIALS_PER_G_VALUE, **system_params_exp2,
            t_vector=T_VECTOR, frequency_s_true=DEFAULT_FREQUENCY_S_TRUE
        )

        # Generate the key string for the JSON dictionary
        # This exact key will be saved in the CSV for perfect matching later
        param_tuple_key_str = str(tuple(v if not np.isinf(v) else "inf" for v in system_params_exp2.values()))

        all_exp2_full_sweep_details[param_tuple_key_str] = {
            'params': system_params_exp2, # Original params dictionary for reference
            'g_lever_values': G_LEVER_VALUES_FOR_SWEEP.tolist(),
            'mi_raw_curve': mi_raw_curve, # List of mean_mi_raw values
            'it_final_curve': df_sweep_results['mean_it_final'].tolist(),
            'tpe_curve': df_sweep_results['mean_tpe'].tolist(),
            'utility_curve': df_sweep_results['mean_utility'].tolist(),
            'cost_curve': df_sweep_results['energetic_cost'].tolist()
        }

        optimal_g_utility, max_utility_value = np.nan, np.nan
        it_final_at_opt, tpe_at_opt, eg_at_opt, mi_raw_at_opt = np.nan, np.nan, np.nan, np.nan

        if not df_sweep_results.empty and 'mean_utility' in df_sweep_results.columns and \
           not df_sweep_results['mean_utility'].isnull().all():
            try:
                optimal_idx = df_sweep_results['mean_utility'].idxmax() # idxmax handles NaNs by ignoring them
                optimal_g_utility = df_sweep_results.loc[optimal_idx, 'g_lever']
                max_utility_value = df_sweep_results.loc[optimal_idx, 'mean_utility']
                it_final_at_opt = df_sweep_results.loc[optimal_idx, 'mean_it_final']
                tpe_at_opt = df_sweep_results.loc[optimal_idx, 'mean_tpe']
                eg_at_opt = df_sweep_results.loc[optimal_idx, 'energetic_cost']
                mi_raw_at_opt = df_sweep_results.loc[optimal_idx, 'mean_mi_raw']
            except ValueError: # Happens if all utilities are NaN
                pass
            except Exception as e:
                print(f" Error finding optimal for combo {i+1}: {e}")


        # Append to list for the summary CSV
        all_exp2_optimal_results_list.append({
            **system_params_exp2,
            'json_sweep_key': param_tuple_key_str,  # <<<--- ADDED THIS KEY
            'optimal_g_utility': optimal_g_utility, 'max_utility': max_utility_value,
            'it_final_at_opt_g': it_final_at_opt, 'tpe_at_opt_g': tpe_at_opt,
            'eg_at_opt_g': eg_at_opt, 'mi_raw_at_opt_g': mi_raw_at_opt
        })

        if (i + 1) % (max(1, len(parameter_combinations_exp2) // 20)) == 0 or (i+1) == len(parameter_combinations_exp2) :
             print(f"\n  Processed Combo {i+1} in {time.time() - start_time_combo:.2f}s. Optimal g_util: {optimal_g_utility:.3f}, Max Util: {max_utility_value:.3f}")

        # Plotting (optional, can be reduced for speed during full runs)
        if len(parameter_combinations_exp2) <= 30 or \
           (i < 5 or i % (max(1,len(parameter_combinations_exp2) // 10)) == 0): # Plot first few and some sparse ones
            try:
                fig_exp2, axs_exp2 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
                param_str_parts = []
                for k_idx, (k,v) in enumerate(system_params_exp2.items()):
                    val_str = f"{v:.2g}" if isinstance(v, float) and not np.isinf(v) else str(v)
                    param_str_parts.append(f"{k}={val_str}")
                    if k_idx % 3 == 2 and k_idx < len(system_params_exp2) -1 : param_str_parts.append("\n")

                param_str_exp2_title = ", ".join(param_str_parts)
                fig_exp2.suptitle(f"Exp2: Metrics vs. Perception Gain for Params:\n{param_str_exp2_title}", fontsize=18)

                axs_exp2[0].plot(df_sweep_results['g_lever'], df_sweep_results['mean_utility'], marker='o', ms=4, label='Utility (U)')
                axs_exp2[0].set_ylabel("Mean Utility")
                if not np.isnan(optimal_g_utility):
                     axs_exp2[0].axvline(optimal_g_utility, color='r', linestyle='--', lw=1, label=f'Optimal g={optimal_g_utility:.2f}')
                axs_exp2[0].legend(fontsize=16)

                axs_exp2[1].plot(df_sweep_results['g_lever'], mi_raw_curve, marker='s', ms=3, linestyle=':', label='MI_raw ($S_{true}$ vs $S_{sat}$)') # Use mi_raw_curve
                axs_exp2[1].plot(df_sweep_results['g_lever'], df_sweep_results['mean_it_final'], marker='.', ms=4, label='IT_final ($S_{true}$ vs $S_{proc}$)')
                if not np.isinf(system_params_exp2['C_proc']):
                    axs_exp2[1].axhline(system_params_exp2['C_proc'], color='grey', linestyle=':', lw=1.5, label=f'C_proc={system_params_exp2["C_proc"]:.1f}')
                axs_exp2[1].set_ylabel("Mean MI / IT (bits)")
                axs_exp2[1].legend(fontsize=16)

                axs_exp2[2].plot(df_sweep_results['g_lever'], df_sweep_results['mean_tpe'], marker='.', ms=4, label='TPE (MSE)')
                axs_exp2[2].set_ylabel("Mean TPE (MSE)")
                if df_sweep_results['mean_tpe'].notna().any() and (df_sweep_results['mean_tpe'] > 0).any():
                    axs_exp2[2].set_yscale('log')

                axs_exp2[3].plot(df_sweep_results['g_lever'], df_sweep_results['energetic_cost'], marker='.', ms=4, label='$E_g$')
                axs_exp2[3].set_ylabel("Energetic Cost")

                efficiency = df_sweep_results['mean_it_final'] / (df_sweep_results['energetic_cost'] + 1e-9)
                axs_exp2[4].plot(df_sweep_results['g_lever'], efficiency, marker='.', ms=4, label='IT / $E_g$ (Efficiency)')
                axs_exp2[4].set_ylabel("Info. Efficiency (IT/$E_g$)")
                axs_exp2[4].set_xlabel("Perception Gain (g_lever)")

                for ax_exp2_single in axs_exp2: ax_exp2_single.grid(True, alpha=0.4, linestyle=':')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_filename = f"exp2_sweep_combo_{i+1}.png"
                plot_path = os.path.join(RESULTS_FOLDER, plot_filename)
                plt.savefig(plot_path)
                plt.close(fig_exp2)
                plt.close('all')
            except Exception as e_plot:
                print(f"  Warning: Could not save plot for combo {i+1}. Error: {e_plot}")


    print("\nSweep complete.")
    df_all_optimal_exp2 = pd.DataFrame(all_exp2_optimal_results_list)
    csv_path_exp2 = os.path.join(RESULTS_FOLDER, "exp2_optimal_g_utility_and_MI_raw_at_opt.csv")
    df_all_optimal_exp2.to_csv(csv_path_exp2, index=False)
    print(f"Experiment 5B optimal results (summary) saved to {csv_path_exp2}")

    full_sweeps_path = os.path.join(RESULTS_FOLDER, "exp2_full_sweeps_data_with_miraw_curves.json")
    with open(full_sweeps_path, 'w') as f:
        json.dump(all_exp2_full_sweep_details, f, indent=2, default=str) # default=str for np types
    print(f"Full sweep data (including MI_raw curves) saved to {full_sweeps_path}")

    total_exp2_time = time.time() - start_time_total_exp2
    print(f"\nTotal Experiment 5B execution time: {total_exp2_time:.2f} seconds ({total_exp2_time/60:.2f} minutes).")
    print("Next step: Run 'analyze_exp2_results.py' on the generated CSV and JSON files.")