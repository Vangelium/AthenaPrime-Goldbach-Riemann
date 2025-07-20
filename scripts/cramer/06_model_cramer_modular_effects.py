

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import curve_fit

# Fitted logarithmic polynomial function parameters for asymptotic trend
# y = a * log(x)**2 + b * log(x) + c
FITTED_ASYMPTOTIC_A = 0.0014
FITTED_ASYMPTOTIC_B = -1.0336
FITTED_ASYMPTOTIC_C = 1.2033

def log_poly_func(x, a, b, c):
    """
    Polynomial function of log(x) for curve fitting: a * log(x)**2 + b * log(x) + c
    """
    log_x = np.log(x)
    return a * log_x**2 + b * log_x + c

def count_distinct_prime_factors(n):
    """
    Calculates omega(n), the number of distinct prime factors of n.
    """
    if n <= 1:
        return 0
    count = 0
    d = 2
    temp_n = n
    while d * d <= temp_n:
        if temp_n % d == 0:
            count += 1
            while temp_n % d == 0:
                temp_n //= d
        d += 1
    if temp_n > 1:
        count += 1
    return count

def constant_func(x, c):
    """
    Constant function for curve fitting: c
    """
    return np.full_like(x, c)

def model_cramer_modular_effects(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure log_p_n is not zero or too small
    df_filtered = df[df['log_p_n'] > 0].copy()
    df_filtered['normalized_error'] = df_filtered['error_term'] / df_filtered['log_p_n']

    # Calculate residual after asymptotic trend removal
    df_filtered['asymptotic_trend_predicted'] = log_poly_func(df_filtered['p_n'], FITTED_ASYMPTOTIC_A, FITTED_ASYMPTOTIC_B, FITTED_ASYMPTOTIC_C)
    df_filtered['residual_after_asymptotic'] = df_filtered['normalized_error'] - df_filtered['asymptotic_trend_predicted']

    print("Calculating omega(p_n)...")
    df_filtered['omega_p_n'] = df_filtered['p_n'].apply(count_distinct_prime_factors)

    # Group by omega_p_n and fit a constant to each band to get omega_fits
    omega_values = sorted(df_filtered['omega_p_n'].unique())
    omega_fits = {}
    for omega_val in omega_values:
        subset = df_filtered[df_filtered['omega_p_n'] == omega_val]
        if len(subset) > 1:
            try:
                params, _ = curve_fit(constant_func, subset['p_n'], subset['residual_after_asymptotic'])
                omega_fits[omega_val] = params[0]
            except RuntimeError:
                omega_fits[omega_val] = np.nan
        else:
            omega_fits[omega_val] = np.nan

    # Calculate residual after removing omega(p_n) dependence
    df_filtered['omega_trend_predicted'] = df_filtered['omega_p_n'].map(omega_fits)
    df_filtered['residual_final'] = df_filtered['residual_after_asymptotic'] - df_filtered['omega_trend_predicted']

    os.makedirs(plots_dir, exist_ok=True)

    # Define modules to analyze and model
    modules_to_model = [5, 6] # Based on your analysis
    modular_corrections = {}

    print("Modeling modular effects...")
    for M in modules_to_model:
        df_filtered[f'p_n_mod_{M}'] = df_filtered['p_n'] % M
        
        # Calculate mean residual for each modular class
        mean_residuals_by_mod = df_filtered.groupby(f'p_n_mod_{M}')['residual_final'].mean()
        modular_corrections[M] = mean_residuals_by_mod.to_dict()
        print(f"  Module {M} corrections: {modular_corrections[M]}")

        # Plotting the mean residuals for each modular class
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(mean_residuals_by_mod.index.astype(str), mean_residuals_by_mod.values, color='skyblue')
        ax.set_title(f'Mean Final Residual by p_n mod {M}', fontsize=16)
        ax.set_xlabel(f'p_n mod {M}', fontsize=12)
        ax.set_ylabel('Mean Residual', fontsize=12)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        plt.tight_layout()
        plot_file_mean_mod = os.path.join(plots_dir, f"08_cramer_mean_residual_mod_{M}_1M.png")
        plt.savefig(plot_file_mean_mod, dpi=300)
        print(f"Plot saved to {plot_file_mean_mod}")
        plt.close(fig)

    # Apply modular corrections to get the final clean residual
    df_filtered['modular_correction'] = 0.0
    for M in modules_to_model:
        # For each prime, find its modular class and apply the corresponding correction
        # We need to ensure that the prime is not 2 or 3 for mod 6, etc.
        # For simplicity, we apply the correction based on the prime itself
        df_filtered['modular_correction'] += df_filtered[f'p_n_mod_{M}'].map(modular_corrections[M])

    df_filtered['final_clean_residual'] = df_filtered['residual_final'] - df_filtered['modular_correction']

    # Plotting the final clean residual
    print("Generating final clean residual plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(df_filtered['p_n'], df_filtered['final_clean_residual'], s=1, alpha=0.5, color='darkgreen')
    ax.set_title('Final Clean Residual Error for Cramér\'s Conjecture (after Modular Effects Removal)', fontsize=18)
    ax.set_xlabel('p_n (Prime Number)', fontsize=14)
    ax.set_ylabel('Final Clean Residual Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    plt.tight_layout()
    plot_file_final_residual = os.path.join(plots_dir, "09_cramer_final_clean_residual_1M.png")
    plt.savefig(plot_file_final_residual, dpi=300)
    print(f"Plot saved to {plot_file_final_residual}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    input_data_file = os.path.join(data_dir, "cramer_data_1000000.csv")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    model_cramer_modular_effects(input_data_file, plots_dir)
    print("\nModeling of Cramér modular effects completed.")
