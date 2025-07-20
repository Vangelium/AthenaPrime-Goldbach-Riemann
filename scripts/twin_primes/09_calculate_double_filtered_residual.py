

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Fitted linear function parameters for omega(x) dependence
# y = a * omega_x + b
FITTED_OMEGA_A = 0.4575
FITTED_OMEGA_B = 11.9139

def omega_x_trend_function(omega_x):
    """
    The fitted linear trend function for omega(x).
    """
    return FITTED_OMEGA_A * omega_x + FITTED_OMEGA_B

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

def calculate_double_filtered_residual(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure x > 1 for log(x) and calculate normalization factor and normalized error
    df_filtered = df[df['x'] > 1].copy()
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    print("Calculating omega(x)...")
    df_filtered['omega_x'] = df_filtered['x'].apply(count_distinct_prime_factors)

    print("Calculating double-filtered residual error...")
    # Calculate the predicted trend based on omega(x)
    df_filtered['predicted_omega_trend'] = df_filtered['omega_x'].apply(omega_x_trend_function)

    # Calculate the double-filtered residual error
    df_filtered['double_filtered_residual'] = df_filtered['normalized_error'] - df_filtered['predicted_omega_trend']

    os.makedirs(plots_dir, exist_ok=True)

    # Plotting the double-filtered residual error
    print("Generating double-filtered residual error plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.scatter(df_filtered['x'], df_filtered['double_filtered_residual'], s=1, alpha=0.5, color='darkred')

    ax.set_title('Double-Filtered Residual Error for Twin Primes (after ω(x) trend removal)', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Double-Filtered Residual Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "09_double_filtered_residual_1M.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    input_data_file = os.path.join(data_dir, "twin_prime_data_1M.csv")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    calculate_double_filtered_residual(input_data_file, plots_dir)
    print("\nDouble-filtered residual calculation and visualization completed.")
