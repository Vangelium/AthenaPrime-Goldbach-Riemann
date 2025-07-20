

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

def model_cramer_omega_dependence(data_file, plots_dir):
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

    os.makedirs(plots_dir, exist_ok=True)

    # Group by omega_p_n and fit a constant to each band
    omega_values = sorted(df_filtered['omega_p_n'].unique())
    omega_fits = {}
    
    print("Fitting constant to each omega(p_n) band...")
    for omega_val in omega_values:
        subset = df_filtered[df_filtered['omega_p_n'] == omega_val]
        if len(subset) > 1: # Need at least 2 points for curve_fit
            try:
                # Fit a constant to the residual for this omega_val
                params, _ = curve_fit(constant_func, subset['p_n'], subset['residual_after_asymptotic'])
                omega_fits[omega_val] = params[0] # The constant value
            except RuntimeError as e:
                print(f"Could not fit constant for omega(p_n)={omega_val}: {e}")
                omega_fits[omega_val] = np.nan
        else:
            omega_fits[omega_val] = np.nan

    # Plotting the omega(p_n) dependence and fits
    print("Generating omega(p_n) dependence plot with fits...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    colors = plt.cm.jet(np.linspace(0, 1, len(omega_values)))
    for i, omega_val in enumerate(omega_values):
        subset = df_filtered[df_filtered['omega_p_n'] == omega_val]
        ax.scatter(subset['p_n'], subset['residual_after_asymptotic'], s=1, alpha=0.5, color=colors[i], label=f'ω(p_n)={omega_val}')
        if not np.isnan(omega_fits[omega_val]):
            ax.axhline(omega_fits[omega_val], color=colors[i], linestyle='--', linewidth=2)

    ax.set_title('Residual Error after Asymptotic Trend Removal, Stratified by ω(p_n)', fontsize=18)
    ax.set_xlabel('p_n (Prime Number)', fontsize=14)
    ax.set_ylabel('Residual Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    ax.legend(title='ω(p_n)')
    plt.tight_layout()
    plot_file_omega_bands = os.path.join(plots_dir, "05_cramer_omega_bands_fit_1M.png")
    plt.savefig(plot_file_omega_bands, dpi=300)
    print(f"Plot saved to {plot_file_omega_bands}")
    plt.close(fig)

    # Calculate and plot the residual error after removing omega(p_n) dependence
    df_filtered['omega_trend_predicted'] = df_filtered['omega_p_n'].map(omega_fits)
    df_filtered['residual_after_omega'] = df_filtered['residual_after_asymptotic'] - df_filtered['omega_trend_predicted']

    print("Generating residual error plot after omega(p_n) dependence removal...")
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(df_filtered['p_n'], df_filtered['residual_after_omega'], s=1, alpha=0.5, color='darkred')
    ax.set_title('Residual Error for Cramér\'s Conjecture (after ω(p_n) Dependence Removal)', fontsize=18)
    ax.set_xlabel('p_n (Prime Number)', fontsize=14)
    ax.set_ylabel('Residual Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    plt.tight_layout()
    plot_file_residual_omega = os.path.join(plots_dir, "06_cramer_residual_after_omega_1M.png")
    plt.savefig(plot_file_residual_omega, dpi=300)
    print(f"Plot saved to {plot_file_residual_omega}")
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

    model_cramer_omega_dependence(input_data_file, plots_dir)
    print("\nModeling of Cramér omega(p_n) dependence completed.")

