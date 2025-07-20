

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

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

def analyze_normalized_error_factors(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure x > 1 for log(x) and calculate normalization factor and normalized error
    df_filtered = df[df['x'] > 1].copy()
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    print("Calculating omega(x) and x mod 6...")
    # Apply the function to calculate omega(x)
    df_filtered['omega_x'] = df_filtered['x'].apply(count_distinct_prime_factors)
    # Calculate x mod 6
    df_filtered['x_mod_6'] = df_filtered['x'] % 6

    os.makedirs(plots_dir, exist_ok=True)

    # --- Visualization 1: Normalized Error vs x, colored by x mod 6 ---
    print("Generating normalized error plot colored by x mod 6...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Define colors for each x_mod_6 value
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange', 5: 'cyan'}
    labels = {0: 'x % 6 == 0', 1: 'x % 6 == 1', 2: 'x % 6 == 2', 3: 'x % 6 == 3', 4: 'x % 6 == 4', 5: 'x % 6 == 5'}

    for mod_val in sorted(df_filtered['x_mod_6'].unique()):
        subset = df_filtered[df_filtered['x_mod_6'] == mod_val]
        ax.scatter(subset['x'], subset['normalized_error'], s=1, alpha=0.5, color=colors.get(mod_val, 'gray'), label=labels.get(mod_val, f'x % 6 == {mod_val}'))

    ax.set_title('Normalized Error Term E₂(x) / (√x / ln x) by x mod 6', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    ax.legend(title='x mod 6')
    plt.tight_layout()
    plot_file_mod6 = os.path.join(plots_dir, "03_normalized_error_by_x_mod_6_1M.png")
    plt.savefig(plot_file_mod6, dpi=300)
    print(f"Plot saved to {plot_file_mod6}")
    plt.close(fig)

    # --- Visualization 2: Normalized Error vs omega(x) ---
    print("Generating normalized error plot vs omega(x)...")
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(df_filtered['omega_x'], df_filtered['normalized_error'], s=1, alpha=0.5, color='darkgreen')
    ax.set_title('Normalized Error Term E₂(x) / (√x / ln x) vs Omega(x)', fontsize=18)
    ax.set_xlabel('Omega(x) (Number of Distinct Prime Factors of x)', fontsize=14)
    ax.set_ylabel('Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    plot_file_omega = os.path.join(plots_dir, "04_normalized_error_vs_omega_x_1M.png")
    plt.savefig(plot_file_omega, dpi=300)
    print(f"Plot saved to {plot_file_omega}")
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

    analyze_normalized_error_factors(input_data_file, plots_dir)
    print("\nAnalysis of normalized error factors completed.")

