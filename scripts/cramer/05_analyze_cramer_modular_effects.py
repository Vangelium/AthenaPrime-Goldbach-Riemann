

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

# Fitted constant values for each omega(p_n) band (from previous run of 04_model_cramer_omega_dependence.py)
# These values need to be updated if the previous script is re-run and values change
# For now, we'll re-calculate them within this script for consistency

def analyze_cramer_modular_effects(data_file, plots_dir):
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

    # Define modules to analyze
    modules = [2, 3, 5, 6, 30]

    print("Analyzing modular effects...")
    for M in modules:
        df_filtered[f'p_n_mod_{M}'] = df_filtered['p_n'] % M

        # Plotting the final residual, colored by modular class
        print(f"Generating residual plot colored by p_n mod {M}...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 10))

        unique_mod_values = sorted(df_filtered[f'p_n_mod_{M}'].unique())
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_mod_values)))

        for i, mod_val in enumerate(unique_mod_values):
            subset = df_filtered[df_filtered[f'p_n_mod_{M}'] == mod_val]
            ax.scatter(subset['p_n'], subset['residual_final'], s=1, alpha=0.5, color=colors[i], label=f'p_n mod {M} == {mod_val}')

        ax.set_title(f'Final Residual Error for Cramér\'s Conjecture (Colored by p_n mod {M})', fontsize=18)
        ax.set_xlabel('p_n (Prime Number)', fontsize=14)
        ax.set_ylabel('Final Residual Error', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
        ax.legend(title=f'p_n mod {M}')
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f"07_cramer_residual_mod_{M}_1M.png")
        plt.savefig(plot_file, dpi=300)
        print(f"Plot saved to {plot_file}")
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

    analyze_cramer_modular_effects(input_data_file, plots_dir)
    print("\nModular effects analysis completed.")
