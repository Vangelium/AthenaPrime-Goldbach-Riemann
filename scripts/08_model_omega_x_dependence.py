

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import curve_fit

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

def linear_func(x, a, b):
    """
    Linear function for curve fitting: a * x + b
    """
    return a * x + b

def model_omega_x_dependence(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure x > 1 for log(x) and calculate normalization factor and normalized error
    df_filtered = df[df['x'] > 1].copy()
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    print("Calculating omega(x)...")
    df_filtered['omega_x'] = df_filtered['x'].apply(count_distinct_prime_factors)

    print("Grouping by omega(x) and calculating mean normalized error...")
    # Calculate mean normalized error for each omega_x value
    omega_grouped_data = df_filtered.groupby('omega_x')['normalized_error'].mean().reset_index()

    # Filter out omega_x values with very few data points if necessary
    # For now, we'll use all, but this might be needed for very high omega_x

    # Perform curve fitting (linear fit as a starting point)
    print("Performing linear curve fit for omega(x) dependence...")
    # Initial guess for a and b
    p0 = [1.0, 0.0] 
    try:
        # Only fit for omega_x > 0, as omega_x=0 is not relevant for primes
        fit_data = omega_grouped_data[omega_grouped_data['omega_x'] > 0]
        if fit_data.empty:
            print("Not enough data points for omega_x > 0 for fitting. Exiting.")
            return

        params, covariance = curve_fit(linear_func, fit_data['omega_x'], fit_data['normalized_error'], p0=p0)
        a_fit, b_fit = params
        print(f"Fitted linear function: y = {a_fit:.4f} * omega_x + {b_fit:.4f}")
    except RuntimeError as e:
        print(f"Error - curve_fit failed: {e}")
        a_fit, b_fit = None, None

    os.makedirs(plots_dir, exist_ok=True)

    # Plotting the omega(x) dependence and fit
    print("Generating omega(x) dependence plot with fit...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Scatter plot of mean error for each omega_x
    ax.scatter(omega_grouped_data['omega_x'], omega_grouped_data['normalized_error'], s=100, color='darkblue', label='Mean Normalized Error per ω(x)')

    if a_fit is not None:
        # Plot the fitted line over the range of omega_x values
        omega_plot = np.linspace(omega_grouped_data['omega_x'].min(), omega_grouped_data['omega_x'].max(), 100)
        ax.plot(omega_plot, linear_func(omega_plot, a_fit, b_fit), color='red', linestyle='--', linewidth=2, label=f'Linear Fit: {a_fit:.4f}*ω(x) + {b_fit:.4f}')

    ax.set_title('Mean Normalized Error vs. ω(x) with Linear Fit', fontsize=18)
    ax.set_xlabel('ω(x) (Number of Distinct Prime Factors of x)', fontsize=14)
    ax.set_ylabel('Mean Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "08_omega_x_dependence_fit_1M.png")
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

    model_omega_x_dependence(input_data_file, plots_dir)
    print("\nModeling of omega(x) dependence completed.")

