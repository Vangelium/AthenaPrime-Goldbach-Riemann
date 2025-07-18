

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import lombscargle, find_peaks

# Parameters from the REFINED model
REFINED_MODEL_PARAMS = {
    'alpha': 0.277923, 'beta': 0.448495, 'c': 0.044996,
    'gamma_0': 0.570737, 'gamma_4': -0.000123
}

def load_riemann_zeros(file_path):
    """Loads a large list of Riemann zeros line by line to conserve memory."""
    print(f"Loading Riemann zeros from {file_path}...")
    zeros = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    zeros.append(float(line.strip()))
                except ValueError:
                    continue # Skip empty or non-float lines
    except FileNotFoundError:
        print(f"Error: Riemann zeros file not found at {file_path}")
        return None
    print(f"Successfully loaded {len(zeros):,} Riemann zeros.")
    return np.array(zeros)

def deeper_anomaly_search(df, params, zeros_file_path, report_path, plot_path):
    """
    Performs the final, deep anomaly search using the refined model, 1M data,
    and a large list of known Riemann zeros.
    """
    # --- 1. Load Zeros and Calculate Residuals ---
    riemann_zeros = load_riemann_zeros(zeros_file_path)
    if riemann_zeros is None: return

    print("Calculating clean residual signal...")
    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])
    p = params
    base_pred = (p['c'] * (df_filtered['N']**p['alpha']) * (df_filtered['omega(N)']**p['beta']))
    mod_effect = np.ones_like(base_pred)
    mod_effect[df_filtered['N'] % 6 == 0] = np.exp(p['gamma_0'])
    mod_effect[df_filtered['N'] % 6 == 4] = np.exp(p['gamma_4'])
    model_pred = base_pred * mod_effect * np.sign(df_filtered['Delta(N)'])
    df_filtered['clean_residual'] = df_filtered['Delta_norm(N)'] - model_pred

    # --- 2. Perform Spectral Analysis ---
    print("Performing high-resolution Lomb-Scargle periodogram...")
    t = df_filtered['N'].values
    y = df_filtered['clean_residual'].values
    max_freq = (riemann_zeros.max() + 5) / (2 * np.pi)
    ang_freqs = np.linspace(0.5 * 2 * np.pi, max_freq, 200000) # Very high resolution
    power = np.array([])
    chunk_size = 500
    num_chunks = int(np.ceil(len(ang_freqs) / chunk_size))
    for i in range(num_chunks):
        chunk_ang_freqs = ang_freqs[i*chunk_size:(i+1)*chunk_size]
        if chunk_ang_freqs.size > 0:
            power = np.concatenate((power, lombscargle(t, y, chunk_ang_freqs, normalize=True)))

    freqs = ang_freqs / (2 * np.pi)

    # --- 3. Find Anomalous Peaks ---
    print("Searching for anomalous peaks...")
    theoretical_freqs = riemann_zeros / (2 * np.pi)
    observed_peaks_indices, _ = find_peaks(power, height=0.01, distance=5)
    observed_peaks_freqs = freqs[observed_peaks_indices]
    
    anomalous_peaks = []
    for i, p_freq in enumerate(observed_peaks_freqs):
        # Check if this peak is near any theoretical zero
        min_dist = np.min(np.abs(theoretical_freqs - p_freq))
        if min_dist > 0.1: # If no known zero is within 0.1 Hz, it's an anomaly
            anomalous_peaks.append({
                'freq': p_freq,
                'power': power[observed_peaks_indices[i]],
                'min_dist_to_known_zero': min_dist
            })

    # --- 4. Generate Report ---
    report = []
    report.append("--- Deep Anomaly Search Report ---")
    report.append(f"Data Source: N up to {df['N'].max():,}")
    report.append(f"Reference Zeros: {len(riemann_zeros):,} loaded from file.")
    report.append(f"Found {len(observed_peaks_indices)} significant peaks in the spectrum.")
    report.append(f"Found {len(anomalous_peaks)} candidate anomalous peaks (min distance > 0.1).")
    report.append("\n--- Top 10 Strongest Anomalous Peaks ---")
    if not anomalous_peaks:
        report.append("No significant anomalous peaks were found.")
    else:
        anomalous_peaks.sort(key=lambda x: x['power'], reverse=True)
        report.append("\n{:<20} {:<15} {:<30}".format('Frequency', 'Power', 'Min Dist to Known Zero'))
        report.append("-" * 65)
        for peak in anomalous_peaks[:10]:
            report.append("{:<20.6f} {:<15.4f} {:<30.6f}".format(peak['freq'], peak['power'], peak['min_dist_to_known_zero']))

    final_report = "\n".join(report)
    print(final_report)
    with open(report_path, 'w') as f:
        f.write(final_report)
    print(f"\nFinal report saved to {report_path}")

    # --- 5. Generate Plot ---
    print("Generating final anomaly plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(freqs, power, color='navy', linewidth=0.5, label='Power Spectrum')
    # Plot anomalous peaks
    if anomalous_peaks:
        anom_freqs = [p['freq'] for p in anomalous_peaks[:10]]
        anom_powers = [p['power'] for p in anomalous_peaks[:10]]
        ax.plot(anom_freqs, anom_powers, 'o', color='magenta', markersize=8, label='Anomalous Peaks')

    ax.set_title('Deep Anomaly Search - Final Power Spectrum (N=1M)', fontsize=18)
    ax.set_xlabel('Frequency (equivalent to γ / 2π)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(freqs.min(), (riemann_zeros[100]/(2*np.pi))) # Zoom into a reasonable range
    ax.set_ylim(0, 0.1) # Zoom on y-axis to see details
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot successfully saved to {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(project_root, 'data', "goldbach_full_analysis_1M.csv")
    zeros_file = os.path.join(os.path.expanduser("~"), "Desktop", "zeros_de_Rimann.txt")
    report_file = os.path.join(project_root, 'analysis_reports', "02_deep_anomaly_search_report.txt")
    plot_file = os.path.join(project_root, 'plots', "12_deep_anomaly_search_1M.png")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        exit()

    df = pd.read_csv(data_file)
    deeper_anomaly_search(df, REFINED_MODEL_PARAMS, zeros_file, report_file, plot_file)
