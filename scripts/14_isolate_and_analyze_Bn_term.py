

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import lombscargle

# Parameters from the REFINED model (Phase 3)
REFINED_MODEL_PARAMS = {
    'alpha': 0.277923, 'beta': 0.448495, 'c': 0.044996,
    'gamma_0': 0.570737, 'gamma_4': -0.000123
}

# First 20 non-trivial zeros for a detailed check
RIEMANN_ZEROS_GAMMA = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 
    40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248, 
    59.347044, 60.831779, 65.085576, 67.079811, 69.546402, 72.067158, 
    75.704691, 77.144840
]

def analyze_bn_term(df, params, output_path):
    """
    Isolates the B_N term from the PMF theory and performs spectral analysis on it.
    """
    # --- 1. Calculate the "Clean Residual" ---
    print("Calculating the clean residual signal...")
    df_filtered = df[(df['N'] > 0) & (df['omega(N)'] > 0)].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])
    p = params
    base_pred = (p['c'] * (df_filtered['N']**p['alpha']) * (df_filtered['omega(N)']**p['beta']))
    mod_effect = np.ones_like(base_pred)
    mod_effect[df_filtered['N'] % 6 == 0] = np.exp(p['gamma_0'])
    mod_effect[df_filtered['N'] % 6 == 4] = np.exp(p['gamma_4'])
    model_pred = base_pred * mod_effect
    df_filtered['clean_residual'] = df_filtered['Delta_norm(N)'] - (model_pred * np.sign(df_filtered['Delta_norm(N)']))

    # --- 2. Isolate the experimental B_N term ---
    print("Isolating the experimental B_N term...")
    # B_N_exp = |Clean_Residual| / sqrt(2^w(N) / ln(N))
    # We take the signed residual to preserve the full signal for spectral analysis
    pmf_denominator = np.sqrt(2**df_filtered['omega(N)'] / np.log(df_filtered['N']))
    df_filtered['B_N_exp'] = df_filtered['clean_residual'] / pmf_denominator

    # --- 3. Perform Spectral Analysis on B_N ---
    print("Performing spectral analysis on the B_N signal...")
    t = df_filtered['N'].values
    y = df_filtered['B_N_exp'].values

    max_freq = (RIEMANN_ZEROS_GAMMA[-1] + 5) / (2 * np.pi)
    ang_freqs = np.linspace(0.5 * 2 * np.pi, max_freq * 2 * np.pi, 50000)

    power = np.array([])
    chunk_size = 500
    num_chunks = int(np.ceil(len(ang_freqs) / chunk_size))
    for i in range(num_chunks):
        chunk_ang_freqs = ang_freqs[i*chunk_size:(i+1)*chunk_size]
        if chunk_ang_freqs.size > 0:
            power = np.concatenate((power, lombscargle(t, y, chunk_ang_freqs, normalize=True)))
        print(f"  Processed spectral chunk {i+1}/{num_chunks}...")

    freqs = ang_freqs / (2 * np.pi)

    # --- 4. Generate the Power Spectrum Plot ---
    print("Generating the B_N power spectrum plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(freqs, power, color='darkviolet', linewidth=0.75)

    expected_freqs = [gamma / (2 * np.pi) for gamma in RIEMANN_ZEROS_GAMMA]
    for i, f_exp in enumerate(expected_freqs):
        ax.axvline(x=f_exp, color='crimson', linestyle='--', linewidth=1, alpha=0.8)

    ax.set_title('Power Spectrum of the Isolated B_N Term (N=1M)', fontsize=18)
    ax.set_xlabel('Frequency (equivalent to γ / 2π)', fontsize=14)
    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(freqs.min(), freqs.max())
    ax.annotate('Spectrum of the isolated quasi-periodic term B_N', 
                xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(project_root, 'data', "goldbach_full_analysis_1M.csv")
    output_plot_file = os.path.join(project_root, 'plots', "15_B_N_term_spectral_analysis.png")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        exit()

    df = pd.read_csv(data_file)
    analyze_bn_term(df, REFINED_MODEL_PARAMS, output_plot_file)
    print("\nB_N term analysis script completed.")
