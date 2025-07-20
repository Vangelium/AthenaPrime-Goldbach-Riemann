

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import curve_fit

def log_poly_func(x, a, b, c):
    """
    Polynomial function of log(x) for curve fitting: a * log(x)**2 + b * log(x) + c
    """
    log_x = np.log(x)
    return a * log_x**2 + b * log_x + c

def model_cramer_asymptotic_trend(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure log_p_n is not zero or too small
    df_filtered = df[df['log_p_n'] > 0].copy()
    df_filtered['normalized_error'] = df_filtered['error_term'] / df_filtered['log_p_n']

    # Perform curve fitting (logarithmic polynomial fit as a starting point)
    print("Performing logarithmic polynomial curve fit for asymptotic trend...")
    # Initial guess for a, b, c
    p0 = [-1.0, 1.0, -1.0] 
    try:
        # Use p_n for fitting, as the trend is against p_n (or log p_n)
        params, covariance = curve_fit(log_poly_func, df_filtered['p_n'], df_filtered['normalized_error'], p0=p0)
        a_fit, b_fit, c_fit = params
        print(f"Fitted logarithmic polynomial function: y = {a_fit:.4f} * (ln p_n)^2 + {b_fit:.4f} * ln p_n + {c_fit:.4f}")
    except RuntimeError as e:
        print(f"Error - curve_fit failed: {e}")
        a_fit, b_fit, c_fit = None, None, None

    os.makedirs(plots_dir, exist_ok=True)

    # Plotting the trend and fit
    print("Generating asymptotic trend plot with fit...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.scatter(df_filtered['p_n'], df_filtered['normalized_error'], s=1, alpha=0.5, color='darkblue', label='Normalized Error')

    if a_fit is not None:
        # Plot the fitted line over the range of p_n values
        p_n_plot = np.linspace(df_filtered['p_n'].min(), df_filtered['p_n'].max(), 500)
        ax.plot(p_n_plot, log_poly_func(p_n_plot, a_fit, b_fit, c_fit), color='red', linestyle='--', linewidth=2, label=f'Log Poly Fit: {a_fit:.4f}(ln p_n)^2 + {b_fit:.4f}ln p_n + {c_fit:.4f}')

    ax.set_title('Normalized Error for Cramér\'s Conjecture with Asymptotic Trend Fit', fontsize=18)
    ax.set_xlabel('p_n (Prime Number)', fontsize=14)
    ax.set_ylabel('Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    ax.legend()
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "03_cramer_asymptotic_trend_fit_1M.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")
    plt.close(fig)

    # Calculate and plot the residual error after removing the asymptotic trend
    if a_fit is not None:
        df_filtered['asymptotic_trend_predicted'] = log_poly_func(df_filtered['p_n'], a_fit, b_fit, c_fit)
        df_filtered['residual_after_asymptotic'] = df_filtered['normalized_error'] - df_filtered['asymptotic_trend_predicted']

        print("Generating residual error plot after asymptotic trend removal...")
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.scatter(df_filtered['p_n'], df_filtered['residual_after_asymptotic'], s=1, alpha=0.5, color='darkgreen')
        ax.set_title('Residual Error for Cramér\'s Conjecture (after Asymptotic Trend Removal)', fontsize=18)
        ax.set_xlabel('p_n (Prime Number)', fontsize=14)
        ax.set_ylabel('Residual Error', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
        plt.tight_layout()
        plot_file_residual = os.path.join(plots_dir, "04_cramer_residual_after_asymptotic_1M.png")
        plt.savefig(plot_file_residual, dpi=300)
    print(f"Plot saved to {plot_file_residual}")
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

    model_cramer_asymptotic_trend(input_data_file, plots_dir)
    print("\nModeling of Cramér asymptotic trend completed.")
