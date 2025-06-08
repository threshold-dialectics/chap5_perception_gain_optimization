#experiment_1.py
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Simulation Parameters ---
N_TIMESTEPS = 1000  # Number of timesteps for each gain value
TRUE_SIGNAL_AMPLITUDE = 1.0
TRUE_SIGNAL_FREQ = 0.05 # cycles per timestep for np.sin(2 * np.pi * freq * t)

# Noise parameters
INHERENT_NOISE_STD = 0.3
GAIN_INDUCED_NOISE_FACTOR = 0.3 
GAIN_INDUCED_NOISE_EXPONENT = 1.5 # std(N_gain) scales with g_lever**exponent

# Saturation parameters
SENSOR_SAT_MIN = -2.0 
SENSOR_SAT_MAX = 2.0

# Downstream processing bottleneck parameters
PROCESSING_CAPACITY = 1.0 # Signal magnitude beyond which processing noise increases
PROCESSING_NOISE_FACTOR = 0.25 # Scales additional processing noise

# Gain lever sweep
G_LEVER_VALUES = np.linspace(0.1, 3.0, 50) # Sweep g_lever 
G_BASELINE = 1.0 # For the 60% rule visualization

# --- Helper Functions ---
def generate_true_signal(n_timesteps, amplitude, freq):
    """Generates a simple sine wave as the true signal."""
    t = np.arange(n_timesteps)
    return amplitude * np.sin(2 * np.pi * freq * t)

def calculate_rmse(estimated, true):
    """Calculates Root Mean Square Error."""
    return np.sqrt(np.mean((estimated - true)**2))

def calculate_snr_db(signal_power_variance, noise_power_variance):
    """Calculates SNR in dB from signal and noise variances."""
    if noise_power_variance <= 1e-9: # Avoid division by zero or log of non-positive
        return np.inf if signal_power_variance > 1e-9 else -np.inf 
    snr = signal_power_variance / noise_power_variance
    return 10 * np.log10(snr) if snr > 1e-9 else -np.inf

# --- Main Simulation Loop ---
results_task_error = []
results_snr_perc = []

s_true = generate_true_signal(N_TIMESTEPS, TRUE_SIGNAL_AMPLITUDE, TRUE_SIGNAL_FREQ)

for g_lever in G_LEVER_VALUES:
    # --- Mechanism 1: Noise Amplification Dominance ---
    # Inherent noise in observation
    n_inherent = np.random.normal(0, INHERENT_NOISE_STD, N_TIMESTEPS)
    
    # Signal components after gain
    amplified_s_true_component = g_lever * s_true
    amplified_n_inherent_component = g_lever * n_inherent
    
    # Gain-induced noise (increases non-linearly with g_lever)
    std_n_gain = GAIN_INDUCED_NOISE_FACTOR * (g_lever**GAIN_INDUCED_NOISE_EXPONENT)
    n_gain = np.random.normal(0, std_n_gain, N_TIMESTEPS)

    # Signal at perceptual stage before saturation (used for SNR_perc calculation)
    s_perc_pre_saturation = amplified_s_true_component + amplified_n_inherent_component + n_gain

    # --- Mechanism 2: Sensor/Channel Saturation ---
    s_saturated = np.clip(s_perc_pre_saturation, SENSOR_SAT_MIN, SENSOR_SAT_MAX)

    # --- Mechanism 3: Downstream Processing Bottlenecks ---
    # Processing noise increases if saturated signal magnitude exceeds capacity
    excess_magnitude = np.maximum(0, np.abs(s_saturated) - PROCESSING_CAPACITY)
    std_n_proc_elementwise = PROCESSING_NOISE_FACTOR * excess_magnitude
    
    n_proc = np.zeros_like(s_saturated)
    # Generate noise element-wise if std varies
    for i in range(N_TIMESTEPS):
        if std_n_proc_elementwise[i] > 1e-9: # Add noise only if std is meaningfully positive
            n_proc[i] = np.random.normal(0, std_n_proc_elementwise[i])
    
    s_final_processed = s_saturated + n_proc

    # --- Calculate Metrics ---
    # Task Performance Error (RMSE)
    task_error = calculate_rmse(s_final_processed, s_true)
    results_task_error.append(task_error)

    # Effective SNR at S_perc 
    # (Signal: amplified true signal; Noise: amplified inherent noise + gain-induced noise)
    # Variances are used as proxies for power, assuming signals/noises might be zero-mean or DC offset is not primary concern for SNR
    power_s_perc_signal_component = np.var(amplified_s_true_component)
    noise_at_perc_stage = amplified_n_inherent_component + n_gain
    power_n_perc_total_component = np.var(noise_at_perc_stage)
    
    snr_perc_db = calculate_snr_db(power_s_perc_signal_component, power_n_perc_total_component)
    results_snr_perc.append(snr_perc_db)

# --- Plotting Results (to match book figure style) ---
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Plot Task Performance Error
color_error = 'tab:red'
ax1.set_xlabel('Perception Gain')
ax1.set_ylabel('Task Performance Error (RMSE)', color=color_error)
ax1.plot(G_LEVER_VALUES, results_task_error, color=color_error, marker='o', markersize=4, linestyle='-', linewidth=1.5, label='Task Performance Error')
ax1.tick_params(axis='y', labelcolor=color_error)
ax1.grid(True, linestyle=':', alpha=0.6)

# Create secondary y-axis for Effective SNR
ax2 = ax1.twinx()
color_snr = 'tab:blue'
ax2.set_ylabel('Effective SNR at S_perc (dB)', color=color_snr) 
ax2.plot(G_LEVER_VALUES, results_snr_perc, color=color_snr, marker='x', markersize=4, linestyle='--', linewidth=1.5, label='Effective SNR at S_perc')
ax2.tick_params(axis='y', labelcolor=color_snr)

# Add 60% rule threshold line (assuming G_BASELINE = 1.0)
threshold_60_percent_val = G_BASELINE * 1.6
ax1.axvline(x=threshold_60_percent_val, color='dimgrey', linestyle='-.', linewidth=1.5, label=f'60% Rule Threshold ({threshold_60_percent_val:.1f})')

# Unified legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)

fig.tight_layout(rect=[0, 0.05, 1, 0.93]) 
plt.title(r'Demonstration of the 60\% Rule: Limits of Perception Gain Enhancement', pad=20) 

# Ensure "Images" directory exists for saving the figure
if not os.path.exists("Images"):
    os.makedirs("Images")
output_figure_path = "Images/Fig_Ch5_60PercentRule_Demo.png" # As per book draft
plt.savefig(output_figure_path, dpi=300)
# print(f"Plot saved to {output_figure_path}")
# plt.show() # For interactive viewing; comment out for script execution for book figures
