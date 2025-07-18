

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def fit_refined_model(df):
    """
    Fits a refined regression model including modular terms.
    log(|Delta_norm|) = log(c) + alpha*log(N) + beta*log(omega) + gamma_0*I(N%6=0) + gamma_4*I(N%6=4)
    """
    print("Preparing data for the refined regression model...")
    df_filtered = df[(df['Delta(N)'] != 0) & (df['N'] > 0)].copy()

    # Create dependent variable: log of normalized, absolute error
    df_filtered['abs_delta_norm'] = np.abs(df_filtered['Delta(N)']) / np.sqrt(df_filtered['N'])
    df_filtered = df_filtered[df_filtered['abs_delta_norm'] > 0]
    y = np.log(df_filtered['abs_delta_norm'])

    # Create dummy variables for N mod 6. N is even, so N%6 can be 0, 2, 4.
    # We use N%6=2 as the baseline category.
    df_filtered['is_mod6_0'] = (df_filtered['N'] % 6 == 0).astype(int)
    df_filtered['is_mod6_4'] = (df_filtered['N'] % 6 == 4).astype(int)

    # Create the design matrix X for the regression
    X = np.vstack([
        np.log(df_filtered['N']), 
        np.log(df_filtered['omega(N)']),
        df_filtered['is_mod6_0'],
        df_filtered['is_mod6_4'],
        np.ones(len(df_filtered)) # for intercept log(c)
    ]).T

    print("Fitting refined model using np.linalg.lstsq...")
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    alpha, beta, gamma_0, gamma_4, log_c = coeffs
    c = np.exp(log_c)
    
    results = {
        'alpha': alpha, 'beta': beta, 'c': c, 
        'gamma_0': gamma_0, 'gamma_4': gamma_4
    }
    return results, df_filtered

def plot_refined_model_fit(df, params, output_path):
    """
    Plots the data and overlays the refined model envelopes, showing splits for mod 6.
    """
    print("Generating plot with refined model overlay...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot the raw normalized data, this time colored by N mod 6 to see the split
    df['N_mod_6'] = df['N'] % 6
    scatter = ax.scatter(
        df['N'], df['Delta(N)'] / np.sqrt(df['N']),
        c=df['N_mod_6'], cmap='plasma', s=8, alpha=0.6, zorder=1
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('N mod 6', fontsize=12)

    # Plot the fitted model envelopes
    n_range = np.linspace(df['N'].min(), df['N'].max(), 500)
    p = params

    for k in sorted(df['omega(N)'].unique()):
        if k < 1: continue
        # Base envelope (for N%6=2)
        base_env = p['c'] * (n_range**p['alpha']) * (k**p['beta'])
        # Envelope for N%6=0
        mod0_env = base_env * np.exp(p['gamma_0'])
        # Envelope for N%6=4
        mod4_env = base_env * np.exp(p['gamma_4'])

        ax.plot(n_range, mod0_env, 'r--', linewidth=1, alpha=0.9, zorder=2)
        ax.plot(n_range, -mod0_env, 'r--', linewidth=1, alpha=0.9, zorder=2)
        ax.plot(n_range, base_env, 'g--', linewidth=1, alpha=0.9, zorder=2)
        ax.plot(n_range, -base_env, 'g--', linewidth=1, alpha=0.9, zorder=2)
        ax.plot(n_range, mod4_env, 'b--', linewidth=1, alpha=0.9, zorder=2)
        ax.plot(n_range, -mod4_env, 'b--', linewidth=1, alpha=0.9, zorder=2)

    ax.set_title('Refined Model (with N mod 6) vs. Normalized Error', fontsize=16)
    ax.set_xlabel('N (Even Numbers)', fontsize=12)
    ax.set_ylabel('Normalized Error Δ(N) / sqrt(N)', fontsize=12)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylim((df['Delta(N)']/np.sqrt(df['N'])).quantile(0.001), (df['Delta(N)']/np.sqrt(df['N'])).quantile(0.999))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='r', lw=2, label='Model (N%6=0)'),
                       Line2D([0], [0], color='g', lw=2, label='Model (N%6=2)'),
                       Line2D([0], [0], color='b', lw=2, label='Model (N%6=4)')]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_file = os.path.join(project_root, 'data', "goldbach_full_analysis_150k.csv")
    output_plot_file = os.path.join(project_root, 'plots', "10_refined_model_fit_150k.png")

    df = pd.read_csv(input_file)
    model_params, df_fit = fit_refined_model(df)

    print("\n--- Refined Model Parameters ---")
    print(f"Exponent alpha (for N): {model_params['alpha']:.6f}")
    print(f"Exponent beta (for omega(N)): {model_params['beta']:.6f}")
    print(f"Coefficient gamma_0 (for N%6=0): {model_params['gamma_0']:.6f}")
    print(f"Coefficient gamma_4 (for N%6=4): {model_params['gamma_4']:.6f}")
    print(f"Base Constant c: {model_params['c']:.6f}")
    print("----------------------------------\n")

    plot_refined_model_fit(df, model_params, output_plot_file)
