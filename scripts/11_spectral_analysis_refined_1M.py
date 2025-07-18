

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import lombscargle

# Parameters from the REFINED model fitted previously
REFINED_MODEL_PARAMS = {
    'alpha': 0.277923,
    'beta': 0.448495,
    'c': 0.044996,
    'gamma_0': 0.570737, # for N%6=0
    'gamma_4': -0.000123  # for N%6=4
}

# First 20 non-trivial zeros for a more detailed check
RIEMANN_ZEROS_GAMMA = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 
    40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248, 
    59.347044, 60.831779, 65.085576, 67.079811, 69.546402, 72.067158, 
    75.704691, 77.144840
]

def final_spectral_analysis(df, params, output_path):
    """
    Performs the final, refined spectral analysis on the 1M dataset.
    """
    print("Calculating clean residual signal using the REFINED model...")
    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])

    # Calculate the refined model's prediction
    p = params
    base_pred = (p['c'] * (df_filtered['N']**p['alpha']) * (df_filtered['omega(N)']**p['beta']))
    mod_effect = np.ones_like(base_pred)
    mod_effect[df_filtered['N'] % 6 == 0] = np.exp(p['gamma_0'])
    mod_effect[df_filtered['N'] % 6 == 4] = np.exp(p['gamma_4'])
    model_pred = base_pred * mod_effect
    model_pred *= np.sign(df_filtered['Delta(N)'])

    df_filtered['clean_residual'] = df_filtered['Delta_norm(N)'] - model_pred

    print("Performing high-resolution Lomb-Scargle periodogram...")
    t = df_filtered['N'].values
    y = df_filtered['clean_residual'].values

    max_freq = (RIEMANN_ZEROS_GAMMA[-1] + 5) / (2 * np.pi)
    ang_freqs = np.linspace(0.5 * 2 * np.pi, max_freq * 2 * np.pi, 50000) # High resolution

    power = np.array([])
    chunk_size = 500
    num_chunks = int(np.ceil(len(ang_freqs) / chunk_size))
    for i in range(num_chunks):
        chunk_ang_freqs = ang_freqs[i*chunk_size:(i+1)*chunk_size]
        if chunk_ang_freqs.size > 0:
            power = np.concatenate((power, lombscargle(t, y, chunk_ang_freqs, normalize=True)))
        print(f"  Processed chunk {i+1}/{num_chunks}...")

    freqs = ang_freqs / (2 * np.pi)

    print("Generating final power spectrum plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(freqs, power, color='navy', linewidth=0.75)

    expected_freqs = [gamma / (2 * np.pi) for gamma in RIEMANN_ZEROS_GAMMA]
    for i, f_exp in enumerate(expected_freqs):
        ax.axvline(x=f_exp, color='crimson', linestyle='--', linewidth=1, alpha=0.8)

    ax.set_title('Final Power Spectrum (Refined Model, N=1M)', fontsize=18)
    ax.set_xlabel('Frequency (equivalent to γ / 2π)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(freqs.min(), freqs.max())
    ax.annotate(f'Data up to N = {df["N"].max():,}\nModel: Refined (with N mod 6)',
                xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_file = os.path.join(project_root, 'data', "goldbach_full_analysis_1M.csv")
    output_plot_file = os.path.join(project_root, 'plots', "11_spectral_analysis_refined_1M.png")

    if not os.path.exists(input_file):
        print(f"Error: Data file not found at {input_file}")
        exit()

    df = pd.read_csv(input_file)
    final_spectral_analysis(df, REFINED_MODEL_PARAMS, output_plot_file)
