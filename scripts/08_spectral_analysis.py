
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import lombscargle

# Parameters from the model fitted in the previous step
# NOTE: These are hardcoded from the output of the previous script for reproducibility.
MODEL_ALPHA = 0.261091
MODEL_BETA = 0.952417
MODEL_C = 0.037392

# First few non-trivial zeros of the Riemann Zeta function (imaginary parts)
# Source: Andrew Odlyzko, http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1.html
RIEMANN_ZEROS_GAMMA = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
]

def spectral_analysis(df, params, output_path):
    """
    Performs spectral analysis on the residual error term after removing the
    fitted model's contribution.
    """
    print("Calculating clean residual signal for spectral analysis...")
    
    # 1. Calculate the actual normalized error
    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])

    # 2. Calculate the model's prediction (the envelope)
    # We need to restore the sign, which was lost in the log-transform regression
    model_pred = (params['c'] * 
                  (df_filtered['N'] ** params['alpha']) * 
                  (df_filtered['omega(N)'] ** params['beta']))
    # The model predicts the magnitude. We assume it follows the same sign as Delta(N).
    model_pred *= np.sign(df_filtered['Delta(N)'])

    # 3. The clean residual is the difference
    df_filtered['clean_residual'] = df_filtered['Delta_norm(N)'] - model_pred

    print("Performing Lomb-Scargle periodogram analysis...")
    # We use N as our time variable. The signal is the clean residual.
    t = df_filtered['N'].values
    y = df_filtered['clean_residual'].values

    # Define the frequency range to scan
    # Frequencies are expected at gamma / (2*pi). Let's scan a relevant range.
    min_freq = 0.5
    max_freq = (RIEMANN_ZEROS_GAMMA[-1] + 5) / (2 * np.pi)
    # The angular frequencies for lombscargle
    ang_freqs = np.linspace(min_freq * 2 * np.pi, max_freq * 2 * np.pi, 10000)

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
        print(f"  Processed chunk {i+1}/{num_chunks}...")

    # Convert angular frequencies back to standard frequencies for plotting
    freqs = ang_freqs / (2 * np.pi)

    print("Generating power spectrum plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(freqs, power, color='blue', linewidth=1)

    # Add vertical lines for the theoretical Riemann zero frequencies
    expected_freqs = [gamma / (2 * np.pi) for gamma in RIEMANN_ZEROS_GAMMA]
    for i, f_exp in enumerate(expected_freqs):
        ax.axvline(x=f_exp, color='red', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'Zero {i+1}' if i == 0 else None)

    ax.set_title('Power Spectrum of Residual Error (Lomb-Scargle Periodogram)', fontsize=16)
    ax.set_xlabel('Frequency (equivalent to γ / 2π)', fontsize=12)
    ax.set_ylabel('Normalized Power', fontsize=12)
    ax.legend(["Lomb-Scargle Power", "Theoretical Riemann Zero Frequencies"])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(freqs.min(), freqs.max())

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')

    input_file = os.path.join(data_dir, "goldbach_full_analysis_150k.csv")
    output_plot_file = os.path.join(plots_dir, "09_spectral_analysis_150k.png")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit()

    df = pd.read_csv(input_file)
    model_params = {'alpha': MODEL_ALPHA, 'beta': MODEL_BETA, 'c': MODEL_C}

    spectral_analysis(df, model_params, output_plot_file)
