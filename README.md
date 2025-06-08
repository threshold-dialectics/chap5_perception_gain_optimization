# Perception Gain Optimization Experiments

This repository contains Python scripts and analysis notebooks for experiments investigating optimal perception gain ($\gLever$) in a simulated signal detection system. The overarching goal is to move beyond fixed heuristics for setting $\gLever$ and towards discovering data-driven, adaptive heuristics that optimize system performance under various conditions.

## Overarching Goal of Experiments (from project description)

To sweep perception gain ($\gLever$) across a wide range relative to a "baseline" or "optimal" operational gain, and observe how various performance metrics and system stability indicators behave. The aim is to find a point or region where increasing $\gLever$ no longer provides proportional benefits or starts to cause degradation, and ultimately to derive adaptive heuristics for setting $\gLever$.

## Experiments

This repository currently includes two main experiments:

### Experiment 1: Signal Detection with Saturation & Gain Noise (Adaptive Heuristic for MSE)

*   **File:** "experiment_1_adaptive_heuristic.py" (previous name might have been "experiment_1.py")
*   **Concept:** An agent estimates a true signal ($S_{true}$) corrupted by inherent input noise ($N_{in}$), amplified by $\gLever$, further corrupted by gain-induced noise ($N_g$), and finally passed through a saturating channel.
*   **Objective:**
    1.  Sweep $\gLever$ for various combinations of system parameters:
        *   Inherent Signal Quality/Noise ($\sigma^2_{in}$)
        *   Gain-Induced Noise Factor ($k_{gain}$)
        *   Gain-Induced Noise Exponent ($\alpha$)
        *   Saturation Threshold ($S_{max}$)
        *   True Signal Amplitude ("amplitude_s_true")
    2.  For each parameter combination, find the "optimal_g_mse" (the $\gLever$ that minimizes Task Performance Error, TPE, measured by Mean Squared Error).
    3.  Train regression models to predict "optimal_g_mse" based on the system parameters, thereby discovering an **adaptive heuristic** for setting $\gLever$.
*   **Key Metrics Measured per $\gLever$ sweep:**
    *   Task Performance Error (TPE): MSE between $S_{true}$ and the final estimate $\hat{S}$.
    *   Effective Signal-to-Noise Ratio (SNR_eff): (This was part of the original Exp1, but the adaptive heuristic discovery focused on MSE).
*   **Output:**
    *   "results_experiment1_adaptive_heuristic/optimal_gLever_sweep_results_final.csv": CSV file containing system parameters and the corresponding "optimal_g_mse" for each combination.
    *   "results_experiment1_adaptive_heuristic/" (subfolder):
        *   Scatter plots showing "optimal_g_mse" vs. individual system parameters.
        *   A pair plot visualizing relationships between features and "optimal_g_mse".
        *   The best trained regression model (e.g., Gradient Boosting or Random Forest) saved as a ".joblib" file.
        *   SHAP summary plot for the best model, showing feature importances.
        *   Contour plot visualizing the decision surface of the best model for its top two features.

### Experiment 2: Information Channel with Energetic Costs & Bottlenecks (Adaptive Heuristic for Utility)

*   **Files:**
    *   "experiment_2_info_cost.py": Generates simulation data.
    *   "analyze_exp2_results.py": Analyzes the data from the first script and builds models.
*   **Concept:** Extends Experiment 1 by:
    1.  Introducing an explicit **energetic cost** for perception gain: $E_g = \kappa \cdot \gLever^{\phi_1}$.
    2.  Modeling a **downstream processing capacity bottleneck** ($C_{proc}$).
    3.  Calculating **Information Throughput (IT)** using Mutual Information (MI) between $S_{true}$ and the processed signal $\hat{S}_{proc}$.
    4.  Optimizing a **Utility/Fitness function** $U = \text{Benefit}(\text{IT, TPE}) - \lambda \cdot E_g$.
*   **Objective:**
    1.  Sweep $\gLever$ for various combinations of system parameters (including $\sigma^2_{in}, k_{gain}, \alpha, S_{max}, C_{proc}, \kappa, \phi_1, \lambda_{cost}$, and utility weights $w_{IT}, w_{TPE}$).
    2.  For each parameter combination, find the "optimal_g_utility" (the $\gLever$ that maximizes the utility function $U$).
    3.  Store detailed sweep data, including "mi_raw" (MI between $S_{true}$ and $S_{saturated}$, before bottleneck) for "Information Saturation Point" analysis.
    4.  Train regression models to predict "optimal_g_utility" based on system parameters, discovering an **adaptive heuristic for utility-optimal gain**.
*   **Key Metrics Measured per $\gLever$ sweep:**
    *   Task Performance Error (TPE)
    *   Information Throughput (IT_final - after bottleneck)
    *   Raw Mutual Information (MI_raw - before bottleneck)
    *   Energetic Cost ($E_g$)
    *   Utility ($U$)
*   **Output ("experiment_2_info_cost.py"):**
    *   "results_experiment2_info_cost/exp2_optimal_g_utility_and_MI_raw_at_opt.csv": CSV with system parameters and the "optimal_g_utility" along with other metrics at that optimum for each combination.
    *   "results_experiment2_info_cost/exp2_full_sweeps_data.json": JSON file containing the *full* sweep data (IT, TPE, Utility, MI_raw curves vs. $\gLever$) for each parameter combination. This is crucial for the "Information Saturation Point" analysis.
    *   "results_experiment2_info_cost/" (subfolder): Sample plots of individual $\gLever$ sweeps.
*   **Output ("analyze_exp2_results.py"):**
    *   "results_experiment2_info_cost/analysis_outputs_v2/":
        *   Scatter plots showing "optimal_g_utility" vs. original and engineered system parameters.
        *   A pair plot visualizing relationships for "optimal_g_utility".
        *   The best trained regression model for predicting "optimal_g_utility" saved as a ".joblib" file.
        *   SHAP summary plot for the best model.
        *   Contour plot visualizing the decision surface of the best model.
        *   (Conceptual sections for Information Saturation Point analysis based on the JSON data).

## Core System Model (Signal Processing Chain)

The basic signal processing model common to both experiments is:
1.  **True Signal:** $S_{true}(t)$ (e.g., sine wave).
2.  **Observed Signal:** $S_{obs}(t) = S_{true}(t) + N_{in}(t)$ (where $N_{in} \sim \mathcal{N}(0, \sigma^2_{in})$).
3.  **Gain Stage:** $S_{gained}(t) = \gLever \cdot S_{obs}(t)$.
4.  **Gain-Induced Noise:** $S_{noisy\_gained}(t) = S_{gained}(t) + N_g(t)$ (where $N_g \sim \mathcal{N}(0, k_{gain} \cdot \gLever^{\alpha} \cdot \sigma^2_{in})$).
5.  **Saturation Stage:** $S_{saturated}(t) = \text{clip}(S_{noisy\_gained}(t), -S_{max}, S_{max})$.
6.  **Experiment 2 - Processing Bottleneck:** $S_{saturated}(t)$ is further processed/quantized to yield $\hat{S}_{proc}(t)$, with information throughput potentially limited by $C_{proc}$.
7.  **Agent's Estimate:** $\hat{S}(t)$ (which is $S_{saturated}(t)$ for Experiment 1, and $\hat{S}_{proc}(t)$ for Experiment 2).

## Requirements

*   Python 3.x
*   NumPy
*   Matplotlib
*   Pandas
*   Scikit-learn
*   Seaborn
*   (Optional but recommended for deeper model interpretation) SHAP: "pip install shap"
*   Joblib (usually installed with scikit-learn)

## How to Run

1.  **Experiment 1 (Adaptive Heuristic for MSE):**
    *   Open "experiment_1_adaptive_heuristic.py".
    *   Adjust "param_grid" if desired (a smaller grid runs faster for testing).
    *   Adjust "NUM_TRIALS_PER_G_VALUE" (higher for more accuracy, lower for speed).
    *   Run the script: "python experiment_1_adaptive_heuristic.py"
    *   Results (CSV, plots, model) will be saved in the "results_experiment1_adaptive_heuristic" folder.

2.  **Experiment 2 (Adaptive Heuristic for Utility):**
    *   **Step 1: Generate Data**
        *   Open "experiment_2_info_cost.py".
        *   Adjust "param_grid_exp2" if desired.
        *   Adjust "NUM_TRIALS_PER_G_VALUE".
        *   Run the script: "python experiment_2_info_cost.py"
        *   This will generate "exp2_optimal_g_utility_and_MI_raw_at_opt.csv" and "exp2_full_sweeps_data.json" in the "results_experiment2_info_cost" folder, along with sample sweep plots.
    *   **Step 2: Analyze Data and Build Models**
        *   Open "analyze_exp2_results.py".
        *   Ensure "CSV_FILE_PATH" points to the CSV generated in Step 1.
        *   Run the script: "python analyze_exp2_results.py"
        *   Analysis results (plots, model) will be saved in the "results_experiment2_info_cost/analysis_outputs_v2/" folder.

## Future Directions and Analysis

The scripts provide a framework for:
*   Exploring wider parameter ranges.
*   Testing different signal types or noise models.
*   Implementing more sophisticated feature engineering.
*   Trying alternative regression models or neural networks for heuristic discovery.
*   Deeper analysis of SHAP interaction values.
*   Systematic analysis of the "Information Saturation Point" using the "mi_raw" data from Experiment 2.
*   Investigating the sensitivity of optimal gain to the utility function's weighting parameters ($w_{IT}, w_{TPE}, \lambda_{cost}$).

## Contribution to Threshold Dialectics

These experiments provide empirical grounding for how the perception gain lever ($\gLever$) should be managed. By moving from fixed heuristics to data-driven adaptive heuristics, they demonstrate a more nuanced and effective approach to optimizing system performance and utility, aligning with the TD principle of dynamic, context-aware adaptation. Experiment 2, in particular, directly incorporates concepts of energetic cost and processing limitations, which are central to the TD framework's understanding of real-world constraints on adaptive capacity.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.