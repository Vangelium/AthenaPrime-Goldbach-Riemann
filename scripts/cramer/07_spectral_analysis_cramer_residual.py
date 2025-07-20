

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

# --- Constants and Helper Functions (from previous scripts) ---

# Fitted logarithmic polynomial function parameters for asymptotic trend
# y = a * log(x)**2 + b * log(x) + c
FITTED_ASYMPTOTIC_A = 0.0014
FITTED_ASYMPTOTIC_B = -1.0336
FITTED_ASYMPTOTIC_C = 1.2033

def log_poly_func(x, a, b, c):
    """
    Polynomial function of log(x) for curve fitting: a * log(x)**2 + b * log(x) + c
    """
    log_x = np.log(x)
    return a * log_x**2 + b * log_x + c

def count_distinct_prime_factors(n):
    """
    Calculates omega(n), the number of distinct prime factors of n.
    """
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

def constant_func(x, c):
    """
    Constant function for curve fitting: c
    """
    return np.full_like(x, c)

# Riemann Zeros (imaginary parts, gamma_k)
# These are the gamma_k values, not gamma_k / (2*pi)
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

# Analysis parameters
FREQ_RANGE_HZ = (0.5, 8.0)  # Range for f = gamma_k / (2*pi)
OVERSAMPLING_FACTOR = 100 # Reduced from 1000*OVERSAMPLING for memory, but still good resolution

def spectral_analysis_cramer_residual(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # --- Re-calculate final clean residual (reusing logic from 06_model_cramer_modular_effects.py) ---
    df_filtered = df[df['log_p_n'] > 0].copy()
    df_filtered['normalized_error'] = df_filtered['error_term'] / df_filtered['log_p_n']

    df_filtered['asymptotic_trend_predicted'] = log_poly_func(df_filtered['p_n'], FITTED_ASYMPTOTIC_A, FITTED_ASYMPTOTIC_B, FITTED_ASYMPTOTIC_C)
    df_filtered['residual_after_asymptotic'] = df_filtered['normalized_error'] - df_filtered['asymptotic_trend_predicted']

    df_filtered['omega_p_n'] = df_filtered['p_n'].apply(count_distinct_prime_factors)

    # Group by omega_p_n and fit a constant to each band to get omega_fits
    omega_values = sorted(df_filtered['omega_p_n'].unique())
    omega_fits = {}
    for omega_val in omega_values:
        subset = df_filtered[df_filtered['omega_p_n'] == omega_val]
        if len(subset) > 1:
            try:
                params, _ = curve_fit(constant_func, subset['p_n'], subset['residual_after_asymptotic'])
                omega_fits[omega_val] = params[0]
            except RuntimeError:
                omega_fits[omega_val] = np.nan
        else:
            omega_fits[omega_val] = np.nan

    df_filtered['omega_trend_predicted'] = df_filtered['omega_p_n'].map(omega_fits)
    df_filtered['residual_final'] = df_filtered['residual_after_asymptotic'] - df_filtered['omega_trend_predicted']

    # Define modules to analyze and model (from 06_model_cramer_modular_effects.py)
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
    # --- End re-calculation of final clean residual ---

    # Prepare data for Lomb-Scargle
    # The time-like variable is log(p_n)
    t = np.log(df_filtered['p_n'].values)
    y = df_filtered['final_clean_residual'].values

    # Remove NaN/Inf values
    valid_indices = np.isfinite(t) & np.isfinite(y)
    t = t[valid_indices]
    y = y[valid_indices]

    if len(t) < 2:
        print("Not enough valid data points for spectral analysis. Exiting.")
        return

    # Define frequencies for Lomb-Scargle
    min_freq = FREQ_RANGE_HZ[0]
    max_freq = FREQ_RANGE_HZ[1]
    num_freqs = int((max_freq - min_freq) * OVERSAMPLING_FACTOR) # Use the defined oversampling factor
    frequencies = np.linspace(min_freq, max_freq, num_freqs)

    print(f"Performing Lomb-Scargle spectral analysis over {len(frequencies)} frequencies...")
    power = lombscargle(t, y, frequencies, normalize=True)

    os.makedirs(plots_dir, exist_ok=True)

    # --- Plotting the full power spectrum ---
    print("Generating full power spectrum plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(frequencies, power, color='blue')
    ax.set_title('Lomb-Scargle Power Spectrum of Final Clean Residual (Cramér)', fontsize=18)
    ax.set_xlabel('Frequency (gamma / 2pi)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Mark expected Riemann frequencies
    for r_freq in RIEMANN_FREQUENCIES:
        if min_freq <= r_freq <= max_freq:
            ax.axvline(r_freq, color='red', linestyle=':', linewidth=0.8, label=f'Riemann Zero (gamma/2pi) = {r_freq:.2f}')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plot_file_full = os.path.join(plots_dir, "10_cramer_power_spectrum_full_1M.png")
    plt.savefig(plot_file_full, dpi=300)
    print(f"Plot saved to {plot_file_full}")
    plt.close(fig)

    # --- Plotting a zoomed-in region ---
    print("Generating zoomed-in power spectrum plot...")
    fig, ax = plt.subplots(figsize=(18, 10))

    zoom_min_freq = 2.0 # Example zoom range
    zoom_max_freq = 4.0 # Example zoom range

    zoom_indices = (frequencies >= zoom_min_freq) & (frequencies <= zoom_max_freq)
    ax.plot(frequencies[zoom_indices], power[zoom_indices], color='blue')
    ax.set_title(f'Lomb-Scargle Power Spectrum (Zoom: {zoom_min_freq}-{zoom_max_freq} Hz)', fontsize=18)
    ax.set_xlabel('Frequency (gamma / 2pi)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Mark expected Riemann frequencies in zoomed region
    for r_freq in RIEMANN_FREQUENCIES:
        if zoom_min_freq <= r_freq <= zoom_max_freq:
            ax.axvline(r_freq, color='red', linestyle=':', linewidth=0.8, label=f'Riemann Zero (gamma/2pi) = {r_freq:.2f}')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plot_file_zoom = os.path.join(plots_dir, "11_cramer_power_spectrum_zoom_1M.png")
    plt.savefig(plot_file_zoom, dpi=300)
    print(f"Plot saved to {plot_file_zoom}")
    plt.close(fig)

    # --- Quantitative Comparison ---
    print("Quantitative comparison with Riemann frequencies:")
    for r_freq in RIEMANN_FREQUENCIES:
        # Find the index of the closest frequency in our computed frequencies
        closest_idx = np.argmin(np.abs(frequencies - r_freq))
        power_at_riemann_freq = power[closest_idx]
        print(f"  Riemann Zero (gamma/2pi) = {r_freq:.4f} | Power = {power_at_riemann_freq:.6f}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    input_data_file = os.path.join(data_dir, "cramer_data_1000000.csv")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    spectral_analysis_cramer_residual(input_data_file, plots_dir)
    print("\nSpectral analysis for Cramér residual completed.")
