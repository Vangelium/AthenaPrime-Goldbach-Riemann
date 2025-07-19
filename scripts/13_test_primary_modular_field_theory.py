

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters from the REFINED model (Phase 3)
REFINED_MODEL_PARAMS = {
    'alpha': 0.277923, 'beta': 0.448495, 'c': 0.044996,
    'gamma_0': 0.570737, 'gamma_4': -0.000123
}

def test_pmf_theory(df, params, plots_dir):
    """
    Tests the Primary Modular Field (PMF) theory by implementing the E(N,w(N))
    formula and verifying the variance scaling hypothesis.
    """
    # --- 1. Calculate the "Clean Residual" from our previous best model ---
    print("Calculating the clean residual signal from the refined model...")
    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])
    p = params
    base_pred = (p['c'] * (df_filtered['N']**p['alpha']) * (df_filtered['omega(N)']**p['beta']))
    mod_effect = np.ones_like(base_pred)
    mod_effect[df_filtered['N'] % 6 == 0] = np.exp(p['gamma_0'])
    mod_effect[df_filtered['N'] % 6 == 4] = np.exp(p['gamma_4'])
    model_pred = base_pred * mod_effect
    df_filtered['clean_residual'] = df_filtered['Delta_norm(N)'] - (model_pred * np.sign(df_filtered['Delta_norm(N)']))

    # --- 2. Implement the new PMF theory formula for E(N,w(N)) ---
    print("Implementing the PMF theory formula E(N,w(N))...")
    # E(N,w(N)) = (-1)^w(N) * sqrt(2^w(N) / ln(N)) * B_N
    # We model the magnitude and assume B_N is a constant B to be estimated.
    # |E| = B * sqrt(2^w(N) / ln(N))
    # Let's estimate B by matching the mean of |clean_residual| with the mean of the formula.
    pmf_term = np.sqrt(2**df_filtered['omega(N)'] / np.log(df_filtered['N']))
    B = np.mean(np.abs(df_filtered['clean_residual'])) / np.mean(pmf_term)
    print(f"Estimated scaling constant B_N for the PMF model: {B:.4f}")
    df_filtered['E_pred'] = B * pmf_term

    # --- 3. Calculate the Second-Order Residual ---
    df_filtered['second_order_residual'] = np.abs(df_filtered['clean_residual']) - df_filtered['E_pred']

    # --- 4. Generate Plot 1: Residual Comparison ---
    print("Generating Plot 1: Residual Comparison...")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16), sharex=True)
    # Plot 1a: Clean Residual (shows omega bands)
    scatter1 = ax1.scatter(df_filtered['N'], df_filtered['clean_residual'], c=df_filtered['omega(N)'], cmap='viridis', s=1, alpha=0.3)
    ax1.set_title('1a: Clean Residual (Error after Phase 3 Model)', fontsize=16)
    ax1.set_ylabel('Error Value')
    ax1.grid(True)
    ax1.set_ylim(np.quantile(df_filtered['clean_residual'], [0.01, 0.99]))
    plt.colorbar(scatter1, ax=ax1, label='ω(N)')
    # Plot 1b: Second-Order Residual (should be random)
    ax2.scatter(df_filtered['N'], df_filtered['second_order_residual'], s=1, alpha=0.3, color='gray')
    ax2.set_title('1b: Second-Order Residual (Error after applying PMF Theory)', fontsize=16)
    ax2.set_xlabel('N', fontsize=12)
    ax2.set_ylabel('Error Value')
    ax2.grid(True)
    ax2.set_ylim(np.quantile(df_filtered['second_order_residual'], [0.01, 0.99]))
    fig1.suptitle('Verification of PMF Theory: Residual Structure Removal', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot1_path = os.path.join(plots_dir, "13_pmf_residual_comparison.png")
    plt.savefig(plot1_path, dpi=300)
    print(f"Plot 1 saved to {plot1_path}")
    plt.close(fig1)

    # --- 5. Verify Variance Scaling Hypothesis ---
    print("Generating Plot 2: Variance Scaling Hypothesis...")
    variance_data = df_filtered.groupby('omega(N)').agg(
        variance=('clean_residual', 'var'),
        mean_log_N=('N', lambda x: np.log(x.mean()))
    ).reset_index()
    variance_data.rename(columns={'omega(N)': 'k'}, inplace=True)
    variance_data['k_div_logN'] = variance_data['k'] / variance_data['mean_log_N']

    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.plot(variance_data['k_div_logN'], variance_data['variance'], 'o-', color='crimson')
    ax.set_title('Verification of Variance Scaling: Var(Δ) vs. k/ln(N)', fontsize=16)
    ax.set_xlabel('k / ln(<N>_k)', fontsize=12)
    ax.set_ylabel('Variance of Clean Residual', fontsize=12)
    ax.grid(True)
    # Fit a line to check for linearity
    m, c = np.polyfit(variance_data['k_div_logN'], variance_data['variance'], 1)
    ax.plot(variance_data['k_div_logN'], m*variance_data['k_div_logN'] + c, '--', color='black', label=f'Linear Fit (R^2={np.corrcoef(variance_data['k_div_logN'], variance_data['variance'])[0,1]**2:.3f})')
    ax.legend()
    plot2_path = os.path.join(plots_dir, "14_pmf_variance_scaling_verification.png")
    plt.savefig(plot2_path, dpi=300)
    print(f"Plot 2 saved to {plot2_path}")
    plt.close(fig2)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(project_root, 'data', "goldbach_full_analysis_1M.csv")
    plots_dir = os.path.join(project_root, 'plots')

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        exit()

    df = pd.read_csv(data_file)
    test_pmf_theory(df, REFINED_MODEL_PARAMS, plots_dir)
    print("\nPMF Theory verification script completed.")

