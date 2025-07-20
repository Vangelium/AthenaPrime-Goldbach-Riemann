

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

# --- Constants and Helper Functions (from previous scripts) ---

# Riemann Zeros (imaginary parts, gamma_k)
RIEMANN_ZEROS_GAMMA = np.array([
    14.134725141734693790457251983562417282807,
    21.022039638771554992628479596908909000000,
    25.010857580145688765000000000000000000000,
    30.424876125930000000000000000000000000000,
    32.935061587739189759000000000000000000000,
    37.586178158825600000000000000000000000000,
    40.918719012147500000000000000000000000000,
    43.327073540000000000000000000000000000000,
    48.005150881167000000000000000000000000000,
    49.773832477000000000000000000000000000000,
    52.970321477000000000000000000000000000000,
    56.446247698000000000000000000000000000000,
    59.347044000000000000000000000000000000000,
    60.831778525000000000000000000000000000000,
    65.112544000000000000000000000000000000000,
    67.079810000000000000000000000000000000000,
    69.546484000000000000000000000000000000000,
    72.067157000000000000000000000000000000000,
    75.704691000000000000000000000000000000000,
    77.446197000000000000000000000000000000000
])

# Convert gamma_k to frequencies f = gamma_k / (2*pi)
RIEMANN_FREQUENCIES = RIEMANN_ZEROS_GAMMA / (2 * np.pi)

# Analysis parameters for Lomb-Scargle
FREQ_RANGE_HZ = (0.5, 8.0)  # Range for f = gamma_k / (2*pi)
OVERSAMPLING_FACTOR = 20 # Further reduced resolution for memory

# --- Helper functions for calculating residuals for each problem ---

# --- Cramér Residual Calculation (from 07_spectral_analysis_cramer_residual.py) ---
# Fitted logarithmic polynomial function parameters for asymptotic trend
FITTED_CRAMER_ASYMPTOTIC_A = 0.0014
FITTED_CRAMER_ASYMPTOTIC_B = -1.0336
FITTED_CRAMER_ASYMPTOTIC_C = 1.2033

def cramer_log_poly_func(x, a, b, c):
    log_x = np.log(x)
    return a * log_x**2 + b * log_x + c

def cramer_count_distinct_prime_factors(n):
    if n <= 1:
        return 0
    count = 0
    d = 2
    temp_n = n
    while d * d <= temp_n:
        if temp_n % d == 0:
            count += 1
            while temp_n % d == 0:
                temp_n //= d
        d += 1
    if temp_n > 1:
        count += 1
    return count

def cramer_constant_func(x, c):
    return np.full_like(x, c)

def get_cramer_final_clean_residual(data_file):
    df = pd.read_csv(data_file)
    df_filtered = df[df['log_p_n'] > 0].copy()
    df_filtered['normalized_error'] = df_filtered['error_term'] / df_filtered['log_p_n']

    df_filtered['asymptotic_trend_predicted'] = cramer_log_poly_func(df_filtered['p_n'], FITTED_CRAMER_ASYMPTOTIC_A, FITTED_CRAMER_ASYMPTOTIC_B, FITTED_CRAMER_ASYMPTOTIC_C)
    df_filtered['residual_after_asymptotic'] = df_filtered['normalized_error'] - df_filtered['asymptotic_trend_predicted']

    df_filtered['omega_p_n'] = df_filtered['p_n'].apply(cramer_count_distinct_prime_factors)

    omega_values = sorted(df_filtered['omega_p_n'].unique())
    omega_fits = {}
    for omega_val in omega_values:
        subset = df_filtered[df_filtered['omega_p_n'] == omega_val]
        if len(subset) > 1:
            try:
                params, _ = curve_fit(cramer_constant_func, subset['p_n'], subset['residual_after_asymptotic'])
                omega_fits[omega_val] = params[0]
            except RuntimeError:
                omega_fits[omega_val] = np.nan
        else:
            omega_fits[omega_val] = np.nan

    df_filtered['omega_trend_predicted'] = df_filtered['omega_p_n'].map(omega_fits)
    df_filtered['residual_final'] = df_filtered['residual_after_asymptotic'] - df_filtered['omega_trend_predicted']

    modules_to_model = [5, 6] 
    modular_corrections = {}
    for M in modules_to_model:
        df_filtered[f'p_n_mod_{M}'] = df_filtered['p_n'] % M
        mean_residuals_by_mod = df_filtered.groupby(f'p_n_mod_{M}')['residual_final'].mean()
        modular_corrections[M] = mean_residuals_by_mod.to_dict()

    df_filtered['modular_correction'] = 0.0
    for M in modules_to_model:
        df_filtered['modular_correction'] += df_filtered[f'p_n_mod_{M}'].map(modular_corrections[M])

    df_filtered['final_clean_residual'] = df_filtered['residual_final'] - df_filtered['modular_correction']
    
    # Return log(p_n) as time-like variable and the final clean residual
    return np.log(df_filtered['p_n'].values), df_filtered['final_clean_residual'].values

# --- Twin Primes Residual Calculation (from AthenaGeminus/scripts/09_double_filtered_residual.py) ---
# Fitted linear function parameters for omega(x) dependence
FITTED_TWIN_OMEGA_A = 0.4575
FITTED_TWIN_OMEGA_B = 11.9139

def twin_omega_x_trend_function(omega_x):
    return FITTED_TWIN_OMEGA_A * omega_x + FITTED_TWIN_OMEGA_B

def twin_count_distinct_prime_factors(n):
    if n <= 1:
        return 0
    count = 0
    d = 2
    temp_n = n
    while d * d <= temp_n:
        if temp_n % d == 0:
            count += 1
            while temp_n % d == 0:
                temp_n //= d
        d += 1
    if temp_n > 1:
        count += 1
    return count

def get_twin_primes_double_filtered_residual(data_file):
    df = pd.read_csv(data_file)
    df_filtered = df[df['x'] > 1].copy()
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    df_filtered['omega_x'] = df_filtered['x'].apply(twin_count_distinct_prime_factors)

    df_filtered['predicted_omega_trend'] = df_filtered['omega_x'].apply(twin_omega_x_trend_function)

    df_filtered['double_filtered_residual'] = df_filtered['normalized_error'] - df_filtered['predicted_omega_trend']
    
    # Return log(x) as time-like variable and the double filtered residual
    return np.log(df_filtered['x'].values), df_filtered['double_filtered_residual'].values

from goldbach_residual_clean import goldbach_residual_clean

def get_goldbach_final_clean_residual(n_max):
    clean_residual, even_numbers = goldbach_residual_clean(n_max)
    # Return log(N) as time-like variable and the clean residual
    return np.log(even_numbers), clean_residual

# --- Cross-Coherence Analysis Function ---
def run_cross_coherence_analysis(cramer_data_path, twin_primes_data_path, goldbach_data_path, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    # Get residuals for each problem
    print("Calculating Cramér residual...")
    t_cramer, y_cramer = get_cramer_final_clean_residual(cramer_data_path)
    print("Calculating Twin Primes residual...")
    t_twin, y_twin = get_twin_primes_double_filtered_residual(twin_primes_data_path)
    print("Calculating Goldbach residual...")
    t_goldbach, y_goldbach = get_goldbach_final_clean_residual(1000000) # Pass n_max

    # Remove NaN/Inf values from all datasets
    valid_cramer = np.isfinite(t_cramer) & np.isfinite(y_cramer)
    t_cramer, y_cramer = t_cramer[valid_cramer], y_cramer[valid_cramer]

    valid_twin = np.isfinite(t_twin) & np.isfinite(y_twin)
    t_twin, y_twin = t_twin[valid_twin], y_twin[valid_twin]

    valid_goldbach = np.isfinite(t_goldbach) & np.isfinite(y_goldbach)
    t_goldbach, y_goldbach = t_goldbach[valid_goldbach], y_goldbach[valid_goldbach]

    # Perform Lomb-Scargle for each residual
    min_freq = FREQ_RANGE_HZ[0]
    max_freq = FREQ_RANGE_HZ[1]
    num_freqs = int((max_freq - min_freq) * OVERSAMPLING_FACTOR)
    frequencies = np.linspace(min_freq, max_freq, num_freqs)

    print("Performing spectral analysis for Cramér...")
    power_cramer = lombscargle(t_cramer, y_cramer, frequencies, normalize=True)
    print("Performing spectral analysis for Twin Primes...")
    power_twin = lombscargle(t_twin, y_twin, frequencies, normalize=True)
    print("Performing spectral analysis for Goldbach...")
    power_goldbach = lombscargle(t_goldbach, y_goldbach, frequencies, normalize=True)

    # --- Plotting all power spectra together ---
    print("Generating comparative power spectrum plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(frequencies, power_cramer, color='blue', label='Cramér Residual')
    ax.plot(frequencies, power_twin, color='green', label='Twin Primes Residual')
    ax.plot(frequencies, power_goldbach, color='orange', linestyle='--', label='Goldbach Residual')

    ax.set_title('Comparative Lomb-Scargle Power Spectra (Cramér, Twin Primes, Goldbach)', fontsize=18)
    ax.set_xlabel('Frequency (gamma / 2pi)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Mark expected Riemann frequencies
    for r_freq in RIEMANN_FREQUENCIES:
        if min_freq <= r_freq <= max_freq:
            ax.axvline(r_freq, color='red', linestyle=':', linewidth=0.8)

    ax.legend()
    plt.tight_layout()
    plot_file_comparison = os.path.join(plots_dir, "12_comparative_power_spectra_1M.png")
    plt.savefig(plot_file_comparison, dpi=300)
    print(f"Plot saved to {plot_file_comparison}")
    plt.close(fig)

    # --- Quantitative Comparison of Power at Riemann Frequencies ---
    print("Quantitative comparison of power at Riemann frequencies:")
    for r_freq in RIEMANN_FREQUENCIES:
        closest_idx = np.argmin(np.abs(frequencies - r_freq))
        power_c = power_cramer[closest_idx]
        power_t = power_twin[closest_idx]
        power_g = power_goldbach[closest_idx]
        print(f"  Riemann Zero (gamma/2pi) = {r_freq:.4f} | Cramér Power = {power_c:.6f} | Twin Power = {power_t:.6f} | Goldbach Power = {power_g:.6f})")

    # --- Quantitative Coherence Metrics ---
    print("\nQuantitative Coherence Metrics:")

    # Pearson Correlation (Cramér vs Twin Primes)
    pearson_corr_cramer_twin = np.corrcoef(power_cramer, power_twin)[0,1]
    print(f"  Pearson Correlation (Cramér vs Twin Primes): {pearson_corr_cramer_twin:.6f}")

    # Spectral Coherence Index (normalized dot product) (Cramér vs Twin Primes)
    coherence_index_cramer_twin = np.sum(power_cramer * power_twin) / (np.linalg.norm(power_cramer) * np.linalg.norm(power_twin))
    print(f"  Spectral Coherence Index (Cramér vs Twin Primes): {coherence_index_cramer_twin:.6f}")

    # --- Pairwise Pearson Correlations ---
    print("\nPairwise Pearson Correlations of Power Spectra:")
    
    # Cramér vs Goldbach
    pearson_corr_cramer_goldbach = np.corrcoef(power_cramer, power_goldbach)[0,1]
    print(f"  Cramér vs Goldbach: {pearson_corr_cramer_goldbach:.6f}")

    # Twin Primes vs Goldbach
    pearson_corr_twin_goldbach = np.corrcoef(power_twin, power_goldbach)[0,1]
    print(f"  Twin Primes vs Goldbach: {pearson_corr_twin_goldbach:.6f}")

    # --- Permutation Tests for Significance ---
    print("\nPermutation Tests (p-values for Pearson Correlation):")

    def permutation_test_correlation(x, y, n_permutations=1000):
        observed_corr = np.corrcoef(x, y)[0, 1]
        permuted_corrs = []
        for _ in range(n_permutations):
            y_permuted = np.random.permutation(y)
            perm_corr = np.corrcoef(x, y_permuted)[0, 1]
            permuted_corrs.append(perm_corr)
        
        # Calculate p-value: proportion of permuted correlations >= observed_corr (for positive correlation)
        p_value = np.sum(np.abs(permuted_corrs) >= np.abs(observed_corr)) / n_permutations
        return p_value

    # Permutation test for Cramér vs Twin Primes
    p_value_cramer_twin = permutation_test_correlation(power_cramer, power_twin)
    print(f"  p-value (Cramér vs Twin Primes): {p_value_cramer_twin:.6f}")

    # Permutation test for Cramér vs Goldbach
    p_value_cramer_goldbach = permutation_test_correlation(power_cramer, power_goldbach)
    print(f"  p-value (Cramér vs Goldbach): {p_value_cramer_goldbach:.6f}")

    # Permutation test for Twin Primes vs Goldbach
    p_value_twin_goldbach = permutation_test_correlation(power_twin, power_goldbach)
    print(f"  p-value (Twin Primes vs Goldbach): {p_value_twin_goldbach:.6f}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cramer_data_dir = os.path.join(project_root, 'data')
    twin_primes_data_dir = os.path.abspath(os.path.join(project_root, '..', 'AthenaGeminus', 'data')) # Path to AthenaGeminus data
    plots_dir = os.path.join(project_root, 'plots')

    cramer_data_file = os.path.join(cramer_data_dir, "cramer_data_1000000.csv")
    twin_primes_data_file = os.path.join(twin_primes_data_dir, "twin_prime_data_1M.csv")
    goldbach_data_file = None # Placeholder, no actual file needed for synthetic data

    if not os.path.exists(cramer_data_file):
        print(f"Error: Cramér data file not found at {cramer_data_file}")
        exit()
    if not os.path.exists(twin_primes_data_file):
        print(f"Error: Twin Primes data file not found at {twin_primes_data_file}")
        exit()

    run_cross_coherence_analysis(cramer_data_file, twin_primes_data_file, goldbach_data_file, plots_dir)
    print("\nCross-coherence analysis completed.")
