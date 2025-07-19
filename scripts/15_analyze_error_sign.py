

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def custom_acf(x, lags):
    """Manual ACF calculation using numpy, robust to zero variance."""
    x_mean = np.mean(x)
    x_var = np.var(x)
    if x_var == 0:
        return np.zeros(lags + 1)
    
    acf = np.zeros(lags + 1)
    acf[0] = 1.0
    for lag in range(1, lags + 1):
        cov = np.mean((x[:-lag] - x_mean) * (x[lag:] - x_mean))
        acf[lag] = cov / x_var
    return acf

def analyze_error_sign(df, plots_dir):
    """
    Performs a multi-faceted analysis of the sign of the error term Delta(N).
    """
    print("Preparing data for sign analysis...")
    df_filtered = df[df['Delta(N)'] != 0].copy()
    df_filtered['sign'] = np.sign(df_filtered['Delta(N)'])

    # --- Plot 1: Global Sign Visualization ---
    print("Generating Plot 1: Global Sign Visualization...")
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    # We plot a subset for clarity, otherwise it's too dense
    subset = df_filtered.sample(n=min(50000, len(df_filtered)), random_state=42).sort_values('N')
    colors = {1: 'blue', -1: 'red'}
    ax1.scatter(subset['N'], subset['sign'], c=subset['sign'].map(colors), s=1, alpha=0.5)
    ax1.set_title('Global Visualization of Error Sign Δ(N) (Sampled)', fontsize=18)
    ax1.set_xlabel('N', fontsize=14)
    ax1.set_ylabel('Sign of Δ(N) (+1 or -1)', fontsize=14)
    ax1.set_yticks([-1, 1])
    ax1.grid(True)
    plot1_path = os.path.join(plots_dir, "16_sign_global_visualization.png")
    plt.savefig(plot1_path, dpi=300)
    print(f"Plot 1 saved to {plot1_path}")
    plt.close(fig1)

    # --- Plot 2: Autocorrelation of the Sign Sequence ---
    print("Generating Plot 2: Autocorrelation...")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    contiguous_sample = df_filtered.head(10000)['sign'].values
    lags = 40
    acf_values = custom_acf(contiguous_sample, lags)
    ax2.stem(range(lags + 1), acf_values)
    ax2.set_title('Autocorrelation of the Error Sign Sequence (First 10,000 values)', fontsize=16)
    ax2.set_xlabel('Lag', fontsize=12)
    ax2.set_ylabel('Autocorrelation', fontsize=12)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    plot2_path = os.path.join(plots_dir, "17_sign_autocorrelation.png")
    plt.savefig(plot2_path, dpi=300)
    print(f"Plot 2 saved to {plot2_path}")
    plt.close(fig2)

    # --- Plot 3: Sign Probability vs. omega(N) ---
    print("Generating Plot 3: Sign Probability vs. omega(N)...")
    prob_by_omega = df_filtered.groupby('omega(N)')['sign'].apply(lambda x: (x > 0).mean()).reset_index()
    prob_by_omega.rename(columns={'sign': 'P(sign > 0)'}, inplace=True)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.bar(prob_by_omega['omega(N)'], prob_by_omega['P(sign > 0)'], color='teal')
    ax3.axhline(0.5, color='red', linestyle='--', label='50% Probability')
    ax3.set_title('Probability of Positive Error Sign vs. ω(N)', fontsize=16)
    ax3.set_xlabel('ω(N) - Number of Distinct Prime Factors', fontsize=12)
    ax3.set_ylabel('P(Δ(N) > 0)', fontsize=12)
    ax3.set_xticks(prob_by_omega['omega(N)'])
    ax3.legend()
    plot3_path = os.path.join(plots_dir, "18_sign_probability_vs_omega.png")
    plt.savefig(plot3_path, dpi=300)
    print(f"Plot 3 saved to {plot3_path}")
    plt.close(fig3)

    # --- Plot 4: Sign Probability vs. N mod 6 ---
    print("Generating Plot 4: Sign Probability vs. N mod 6...")
    df_filtered['N_mod_6'] = df_filtered['N'] % 6
    prob_by_mod6 = df_filtered.groupby('N_mod_6')['sign'].apply(lambda x: (x > 0).mean()).reset_index()
    prob_by_mod6.rename(columns={'sign': 'P(sign > 0)'}, inplace=True)
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.bar(prob_by_mod6['N_mod_6'].astype(str), prob_by_mod6['P(sign > 0)'], color='purple')
    ax4.axhline(0.5, color='red', linestyle='--', label='50% Probability')
    ax4.set_title('Probability of Positive Error Sign vs. N mod 6', fontsize=16)
    ax4.set_xlabel('N mod 6', fontsize=12)
    ax4.set_ylabel('P(Δ(N) > 0)', fontsize=12)
    ax4.legend()
    plot4_path = os.path.join(plots_dir, "19_sign_probability_vs_mod6.png")
    plt.savefig(plot4_path, dpi=300)
    print(f"Plot 4 saved to {plot4_path}")
    plt.close(fig4)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(project_root, 'data', "goldbach_full_analysis_1M.csv")
    plots_dir = os.path.join(project_root, 'plots')

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        exit()

    df = pd.read_csv(data_file)
    analyze_error_sign(df, plots_dir)
    print("\nSign analysis script completed.")
