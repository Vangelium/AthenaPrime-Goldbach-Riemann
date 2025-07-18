

import pandas as pd
import numpy as np
from scipy.signal import lombscargle, find_peaks
import os

# Parameters from the model fitted previously
MODEL_ALPHA = 0.261091
MODEL_BETA = 0.952417
MODEL_C = 0.037392

# First 10 non-trivial zeros of the Riemann Zeta function (imaginary parts)
RIEMANN_ZEROS_GAMMA = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
]

def quantify_spectral_peaks():
    """
    Performs spectral analysis and quantifies the alignment of power spectrum
    peaks with theoretical Riemann zero frequencies.
    """
    # --- 1. Load Data and Calculate Clean Residual ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_file = os.path.join(project_root, 'data', "goldbach_full_analysis_150k.csv")
    if not os.path.exists(input_file):
        return "Error: Input data file not found."

    df = pd.read_csv(input_file)
    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])
    model_pred = (MODEL_C * (df_filtered['N'] ** MODEL_ALPHA) * 
                  (df_filtered['omega(N)'] ** MODEL_BETA)) * np.sign(df_filtered['Delta(N)'])
    df_filtered['clean_residual'] = df_filtered['Delta_norm(N)'] - model_pred

    t = df_filtered['N'].values
    y = df_filtered['clean_residual'].values

    # --- 2. Perform Lomb-Scargle Periodogram ---
    max_freq = (RIEMANN_ZEROS_GAMMA[-1] + 5) / (2 * np.pi)
    ang_freqs = np.linspace(0.5 * 2 * np.pi, max_freq * 2 * np.pi, 20000) # Higher resolution

    # Calculate the periodogram in chunks to save memory
    power = np.array([])
    chunk_size = 1000 # Process 1000 frequencies at a time
    num_chunks = int(np.ceil(len(ang_freqs) / chunk_size))

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_ang_freqs = ang_freqs[start_idx:end_idx]
        
        if chunk_ang_freqs.size == 0:
            continue

        chunk_power = lombscargle(t, y, chunk_ang_freqs, normalize=True)
        power = np.concatenate((power, chunk_power))

    freqs = ang_freqs / (2 * np.pi)

    # --- 3. Find and Analyze Peaks ---
    report = []
    report.append("--- Quantitative Analysis of Spectral Peaks ---")
    report.append("Comparing theoretical Riemann frequencies with observed peaks in the Goldbach error spectrum.")
    report.append("\n{:<5} {:<20} {:<20} {:<20} {:<15}".format('Zero', 'Theoretical Freq', 'Observed Freq', 'Difference', 'Power'))
    report.append("-" * 80)

    theoretical_freqs = [gamma / (2 * np.pi) for gamma in RIEMANN_ZEROS_GAMMA]
    observed_peaks_indices, _ = find_peaks(power, height=0.005) # Find all significant peaks
    observed_peaks_freqs = freqs[observed_peaks_indices]
    observed_peaks_power = power[observed_peaks_indices]

    found_peak_indices = set()

    for i, th_freq in enumerate(theoretical_freqs):
        # Find the closest observed peak within a reasonable window (e.g., 0.5 Hz)
        window = 0.5
        nearby_peaks_mask = np.abs(observed_peaks_freqs - th_freq) < window
        
        if np.any(nearby_peaks_mask):
            nearby_powers = observed_peaks_power[nearby_peaks_mask]
            strongest_peak_idx_local = np.argmax(nearby_powers)
            
            # Get the original index of the strongest peak
            strongest_peak_freq = observed_peaks_freqs[nearby_peaks_mask][strongest_peak_idx_local]
            strongest_peak_power = nearby_powers[strongest_peak_idx_local]
            diff = strongest_peak_freq - th_freq

            # Find the original index to mark it as found
            original_idx = np.where(observed_peaks_freqs == strongest_peak_freq)[0][0]
            found_peak_indices.add(observed_peaks_indices[original_idx])

            report.append("{:<5} {:<20.6f} {:<20.6f} {:<20.6f} {:<15.4f}".format(
                i + 1, th_freq, strongest_peak_freq, diff, strongest_peak_power))
        else:
            report.append("{:<5} {:<20.6f} {:<20} {:<20} {:<15}".format(
                i + 1, th_freq, "No peak found", "N/A", "N/A"))

    # --- 4. Search for Anomalous Peaks ---
    all_peak_indices = set(observed_peaks_indices)
    anomalous_indices = list(all_peak_indices - found_peak_indices)
    anomalous_freqs = freqs[anomalous_indices]
    anomalous_powers = power[anomalous_indices]

    report.append("\n--- Search for Anomalous Peaks ---")
    if not anomalous_freqs.any():
        report.append("No significant anomalous peaks found.")
    else:
        # Sort by power to find the strongest anomaly
        strongest_anomaly_idx = np.argmax(anomalous_powers)
        anomaly_freq = anomalous_freqs[strongest_anomaly_idx]
        anomaly_power = anomalous_powers[strongest_anomaly_idx]
        report.append("Strongest peak NOT associated with the first 10 Riemann zeros:")
        report.append("Frequency: {:.6f}, Power: {:.4f}".format(anomaly_freq, anomaly_power))

    return "\n".join(report)

if __name__ == "__main__":
    analysis_report = quantify_spectral_peaks()
    print(analysis_report)

    # Save the report to a file for documentation
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'analysis_reports')
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "01_spectral_peak_analysis_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(analysis_report)
    print(f"\nReport saved to {report_file}")

