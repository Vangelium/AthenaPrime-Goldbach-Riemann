

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters from the original REFINED model (trained on all 1M data)
# for comparison.
ORIGINAL_MODEL_PARAMS = {
    'alpha': 0.277923, 'beta': 0.448495, 'c': 0.044996,
    'gamma_0': 0.570737, 'gamma_4': -0.000123
}

def fit_model_on_subset(df_subset):
    """Fits the refined regression model on a subset of the data."""
    df_filtered = df_subset[(df_subset['Delta(N)'] != 0) & (df_subset['N'] > 0)].copy()
    df_filtered['abs_delta_norm'] = np.abs(df_filtered['Delta(N)']) / np.sqrt(df_filtered['N'])
    df_filtered = df_filtered[df_filtered['abs_delta_norm'] > 0]
    y = np.log(df_filtered['abs_delta_norm'])

    df_filtered['is_mod6_0'] = (df_filtered['N'] % 6 == 0).astype(int)
    df_filtered['is_mod6_4'] = (df_filtered['N'] % 6 == 4).astype(int)

    X = np.vstack([
        np.log(df_filtered['N']), np.log(df_filtered['omega(N)']),
        df_filtered['is_mod6_0'], df_filtered['is_mod6_4'],
        np.ones(len(df_filtered))
    ]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta, gamma_0, gamma_4, log_c = coeffs
    return {'alpha': alpha, 'beta': beta, 'c': np.exp(log_c), 'gamma_0': gamma_0, 'gamma_4': gamma_4}

def cross_validate_model(df, plots_dir):
    """
    Performs cross-validation by training on the first half of the data
    and testing on the second half.
    """
    print("Splitting data into training (N<=500k) and testing (N>500k) sets...")
    train_df = df[df['N'] <= 500000].copy()
    test_df = df[df['N'] > 500000].copy()

    print("Training model on the first 500k data points...")
    trained_params = fit_model_on_subset(train_df)

    # --- Generate Report ---
    print("\n--- Model Parameter Stability Report ---")
    print("{:<12} {:<20} {:<20}".format('Parameter', 'Original (1M data)', 'Trained (500k data)'))
    print("-" * 55)
    for key in ORIGINAL_MODEL_PARAMS:
        print("{:<12} {:<20.6f} {:<20.6f}".format(key, ORIGINAL_MODEL_PARAMS[key], trained_params[key]))

    # --- Generate Plot ---
    print("\nGenerating cross-validation plot...")
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plot the unseen test data
    test_df['Delta_norm(N)'] = test_df['Delta(N)'] / np.sqrt(test_df['N'])
    ax.scatter(test_df['N'], test_df['Delta_norm(N)'], s=1, alpha=0.2, color='gray', label='Unseen Test Data (N>500k)')

    # Plot the predictive envelopes from the trained model
    n_range = test_df['N'].values
    p = trained_params
    for k in sorted(test_df['omega(N)'].unique()):
        if k < 1: continue
        base_env = p['c'] * (n_range**p['alpha']) * (k**p['beta'])
        mod0_env = base_env * np.exp(p['gamma_0'])
        ax.plot(n_range[test_df['N'] % 6 == 0], mod0_env[test_df['N'] % 6 == 0], 'r--', linewidth=1)
        ax.plot(n_range[test_df['N'] % 6 == 0], -mod0_env[test_df['N'] % 6 == 0], 'r--', linewidth=1)

    ax.set_title('Model Cross-Validation: Predicting N>500k from model trained on N<=500k', fontsize=18)
    ax.set_xlabel('N', fontsize=14)
    ax.set_ylabel('Normalized Error Δ(N) / sqrt(N)', fontsize=14)
    ax.grid(True)
    ax.legend()
    ax.set_ylim(test_df['Delta_norm(N)'].quantile(0.01), test_df['Delta_norm(N)'].quantile(0.99))

    plot_path = os.path.join(plots_dir, "20_cross_validation_robustness_test.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(project_root, 'data', "goldbach_full_analysis_1M.csv")
    plots_dir = os.path.join(project_root, 'plots')

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        exit()

    df = pd.read_csv(data_file)
    cross_validate_model(df, plots_dir)
    print("\nCross-validation script completed.")

