

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import lombscargle

# Fitted logarithmic function parameters from previous step
# y = a * log(x) + b
FITTED_A = 3.1904
FITTED_B = -27.8122

def log_trend_function(x):
    """
    The fitted logarithmic trend function.
    """
    return FITTED_A * np.log(x) + FITTED_B

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
OVERSAMPLING = 10

def spectral_analysis_residual_error(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure x > 1 for log(x) and calculate normalization factor and normalized error
    df_filtered = df[df['x'] > 1].copy()
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    # Calculate x mod 6
    df_filtered['x_mod_6'] = df_filtered['x'] % 6

    # Filter for x_mod_6 == 5
    df_mod6_5 = df_filtered[df_filtered['x_mod_6'] == 5].copy()

    if df_mod6_5.empty:
        print("No data found for x % 6 == 5. Exiting.")
        return

    print("Calculating residual error...")
    # Calculate the predicted trend for x_mod_6 == 5 points
    df_mod6_5['predicted_trend'] = df_mod6_5['x'].apply(log_trend_function)

    # Calculate the residual error
    df_mod6_5['residual_error'] = df_mod6_5['normalized_error'] - df_mod6_5['predicted_trend']

    # Prepare data for Lomb-Scargle
    # The time-like variable is log(x)
    t = np.log(df_mod6_5['x'].values)
    y = df_mod6_5['residual_error'].values

    # Remove NaN/Inf values that might arise from log(x) for x <= 1 or other issues
    valid_indices = np.isfinite(t) & np.isfinite(y)
    t = t[valid_indices]
    y = y[valid_indices]

    if len(t) < 2:
        print("Not enough valid data points for spectral analysis. Exiting.")
        return

    # Define frequencies for Lomb-Scargle
    # Frequencies are in units of 1/log(x), which corresponds to gamma / (2*pi)
    min_freq = FREQ_RANGE_HZ[0]
    max_freq = FREQ_RANGE_HZ[1]
    num_freqs = int((max_freq - min_freq) * 100) # Reduced resolution for memory
    frequencies = np.linspace(min_freq, max_freq, num_freqs)

    print(f"Performing Lomb-Scargle spectral analysis over {len(frequencies)} frequencies...")
    # Normalize power to be between 0 and 1
    power = lombscargle(t, y, frequencies, normalize=True)

    os.makedirs(plots_dir, exist_ok=True)

    # --- Plotting the full power spectrum ---
    print("Generating full power spectrum plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(frequencies, power, color='blue')
    ax.set_title('Lomb-Scargle Power Spectrum of Residual Error (x mod 6 == 5)', fontsize=18)
    ax.set_xlabel('Frequency (γ / 2π)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Mark expected Riemann frequencies
    for r_freq in RIEMANN_FREQUENCIES:
        if min_freq <= r_freq <= max_freq:
            ax.axvline(r_freq, color='red', linestyle=':', linewidth=0.8, label=f'Riemann Zero (γ/2π) = {r_freq:.2f}')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plot_file_full = os.path.join(plots_dir, "08_power_spectrum_full_1M.png")
    plt.savefig(plot_file_full, dpi=300)
    print(f"Plot saved to {plot_file_full}")
    plt.close(fig)

    # --- Plotting a zoomed-in region ---
    print("Generating zoomed-in power spectrum plot...")
    fig, ax = plt.subplots(figsize=(18, 10))

    zoom_min_freq = 2.0 # As suggested by user
    zoom_max_freq = 4.0 # As suggested by user

    # Filter frequencies and power for the zoomed region
    zoom_indices = (frequencies >= zoom_min_freq) & (frequencies <= zoom_max_freq)
    ax.plot(frequencies[zoom_indices], power[zoom_indices], color='blue')
    ax.set_title(f'Lomb-Scargle Power Spectrum (Zoom: {zoom_min_freq}-{zoom_max_freq} Hz)', fontsize=18)
    ax.set_xlabel('Frequency (γ / 2π)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Mark expected Riemann frequencies in zoomed region
    for r_freq in RIEMANN_FREQUENCIES:
        if zoom_min_freq <= r_freq <= zoom_max_freq:
            ax.axvline(r_freq, color='red', linestyle=':', linewidth=0.8, label=f'Riemann Zero (γ/2π) = {r_freq:.2f}')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plot_file_zoom = os.path.join(plots_dir, "09_power_spectrum_zoom_1M.png")
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

    input_data_file = os.path.join(data_dir, "twin_prime_data_1M.csv")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    spectral_analysis_residual_error(input_data_file, plots_dir)
    print("\nSpectral analysis completed.")
