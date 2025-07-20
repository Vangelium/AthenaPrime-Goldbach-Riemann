

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Fitted logarithmic function parameters from previous step
# y = a * log(x) + b
FITTED_A = 3.1904
FITTED_B = -27.8122

def log_trend_function(x):
    """
    The fitted logarithmic trend function.
    """
    return FITTED_A * np.log(x) + FITTED_B

def visualize_residual_error(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure x > 1 for log(x) and calculate normalization factor and normalized error
    df_filtered = df[df['x'] > 1].copy()
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    # Calculate x mod 6
    df_filtered['x_mod_6'] = df_filtered['x'] % 6

    # Filter for x_mod_6 == 5
    df_mod6_5 = df_filtered[df_filtered['x_mod_6'] == 5].copy()

    if df_mod6_5.empty:
        print("No data found for x % 6 == 5. Exiting.")
        return

    print("Calculating residual error...")
    # Calculate the predicted trend for x_mod_6 == 5 points
    df_mod6_5['predicted_trend'] = df_mod6_5['x'].apply(log_trend_function)

    # Calculate the residual error
    df_mod6_5['residual_error'] = df_mod6_5['normalized_error'] - df_mod6_5['predicted_trend']

    os.makedirs(plots_dir, exist_ok=True)

    # Plotting the residual error
    print("Generating residual error plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.scatter(df_mod6_5['x'], df_mod6_5['residual_error'], s=1, alpha=0.5, color='purple')

    ax.set_title('Residual Error for Twin Primes (x mod 6 == 5) after Log Trend Removal', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Residual Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "07_residual_error_x_mod_6_5_1M.png")
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

    visualize_residual_error(input_data_file, plots_dir)
    print("\nResidual error visualization completed.")
