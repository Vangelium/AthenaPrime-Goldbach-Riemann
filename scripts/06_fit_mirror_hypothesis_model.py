

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def fit_mirror_hypothesis_model(df):
    """
    Fits a multiple linear regression model to test the Mirror Hypothesis:
    log(|Delta(N)|) = log(c) + alpha*log(N) + beta*log(omega(N))

    Args:
        df (pd.DataFrame): DataFrame with N, Delta(N), and omega(N).

    Returns:
        dict: A dictionary containing the fitted parameters alpha, beta, and c.
    """
    print("Preparing data for regression model...")
    # Filter out rows where Delta(N) is zero to avoid log(0) issues.
    df_filtered = df[df['Delta(N)'] != 0].copy()
    
    # It's better to model the normalized error to be consistent with our plots
    # |Delta(N)|/sqrt(N) = c * N^alpha * omega(N)^beta
    # log(|Delta(N)|/sqrt(N)) = log(c) + alpha*log(N) + beta*log(omega(N))
    
    df_filtered['abs_delta_norm'] = np.abs(df_filtered['Delta(N)']) / np.sqrt(df_filtered['N'])
    df_filtered = df_filtered[df_filtered['abs_delta_norm'] > 0]

    y = np.log(df_filtered['abs_delta_norm'])
    X = np.vstack([
        np.log(df_filtered['N']), 
        np.log(df_filtered['omega(N)']),
        np.ones(len(df_filtered)) # for intercept log(c)
    ]).T

    print("Fitting model using np.linalg.lstsq...")
    # Use NumPy's least squares to find the best fit for the coefficients
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    alpha, beta, log_c = coeffs
    c = np.exp(log_c)
    
    results = {'alpha': alpha, 'beta': beta, 'c': c}
    return results

def plot_model_fit(df, params, output_path):
    """
    Plots the normalized error data and overlays the fitted model envelopes.
    """
    print("Generating plot with fitted model overlay...")
    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot the raw normalized data, colored by omega(N)
    scatter = ax.scatter(
        df_filtered['N'], df_filtered['Delta_norm(N)'], c=df_filtered['omega(N)'],
        cmap='viridis', s=8, alpha=0.6, zorder=1
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ω(N) - Number of Distinct Prime Factors', fontsize=12)

    # Plot the fitted model envelopes
    n_range = np.linspace(df_filtered['N'].min(), df_filtered['N'].max(), 500)
    alpha, beta, c = params['alpha'], params['beta'], params['c']

    for k in sorted(df_filtered['omega(N)'].unique()):
        if k < 1: continue
        # Calculate envelope: c * N^alpha * k^beta
        envelope = c * (n_range ** alpha) * (k ** beta)
        ax.plot(n_range, envelope, 'r--', linewidth=1.5, alpha=0.8, zorder=2)
        ax.plot(n_range, -envelope, 'r--', linewidth=1.5, alpha=0.8, zorder=2)

    ax.set_title('Fitted Model vs. Normalized Error (Mirror Hypothesis Test)', fontsize=16)
    ax.set_xlabel('N (Even Numbers)', fontsize=12)
    ax.set_ylabel('Normalized Error Δ(N) / sqrt(N)', fontsize=12)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylim(df_filtered['Delta_norm(N)'].min() * 1.1, df_filtered['Delta_norm(N)'].max() * 1.1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')

    input_file = os.path.join(data_dir, "goldbach_full_analysis_150k.csv")
    output_plot_file = os.path.join(plots_dir, "04_mirror_hypothesis_fit_150k.png")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit()

    df = pd.read_csv(input_file)
    
    # Fit the model and get parameters
    model_params = fit_mirror_hypothesis_model(df)
    
    # Print results
    alpha = model_params['alpha']
    beta = model_params['beta']
    hypothesis_sum = alpha + beta

    print("\n--- Mirror Hypothesis Model Results ---")
    print(f"Exponent alpha (for N): {alpha:.6f}")
    print(f"Exponent beta (for omega(N)): {beta:.6f}")
    print(f"Constant c: {model_params['c']:.6f}")
    print("-----------------------------------------")
    print(f"Sum of exponents (alpha + beta): {hypothesis_sum:.6f}")
    print("-----------------------------------------\n")

    # Generate the plot
    plot_model_fit(df, model_params, output_plot_file)
