

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import curve_fit

def log_func(x, a, b):
    """
    Logarithmic function for curve fitting: a * log(x) + b
    """
    return a * np.log(x) + b

def fit_mod6_5_trend(data_file, plots_dir):
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

    print("Binning data for trend fitting...")
    # Define bins for x to calculate mean error in intervals
    num_bins = 100 # Increased granularity for fitting
    bins = np.linspace(df_mod6_5['x'].min(), df_mod6_5['x'].max(), num_bins)
    df_mod6_5['x_bin'] = pd.cut(df_mod6_5['x'], bins, include_lowest=True)

    # Calculate mean normalized error for each bin
    binned_data = df_mod6_5.groupby('x_bin')['normalized_error'].mean().reset_index()
    binned_data['x_mid'] = binned_data['x_bin'].apply(lambda x: x.mid).astype(float)
    binned_data = binned_data.dropna() # Drop bins with no data

    # Perform curve fitting
    print("Performing logarithmic curve fit...")
    # Initial guess for a and b (can be refined if fit is poor)
    p0 = [1.0, 1.0] 
    try:
        params, covariance = curve_fit(log_func, binned_data['x_mid'], binned_data['normalized_error'], p0=p0)
        a_fit, b_fit = params
        print(f"Fitted logarithmic function: y = {a_fit:.4f} * log(x) + {b_fit:.4f}")
    except RuntimeError as e:
        print(f"Error - curve_fit failed: {e}")
        a_fit, b_fit = None, None

    os.makedirs(plots_dir, exist_ok=True)

    # Plotting the trend and fit
    print("Generating trend and fit plot for x % 6 == 5...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.scatter(df_mod6_5['x'], df_mod6_5['normalized_error'], s=1, alpha=0.1, color='darkcyan', label='Data points (x % 6 == 5)')
    ax.plot(binned_data['x_mid'], binned_data['normalized_error'], color='red', linewidth=2, label='Mean Normalized Error (Binned)')
    
    if a_fit is not None:
        x_plot = np.linspace(binned_data['x_mid'].min(), binned_data['x_mid'].max(), 500)
        ax.plot(x_plot, log_func(x_plot, a_fit, b_fit), color='green', linestyle='--', linewidth=2, label=f'Log Fit: {a_fit:.4f}*ln(x) + {b_fit:.4f}')

    ax.set_title('Normalized Error Term E₂(x) / (√x / ln x) for x mod 6 == 5 with Log Fit', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    ax.legend()
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "06_normalized_error_log_fit_x_mod_6_5_1M.png")
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

    fit_mod6_5_trend(input_data_file, plots_dir)
    print("\nLogarithmic fit for x mod 6 == 5 trend completed.")

