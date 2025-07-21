import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

# Fitted linear function parameters for omega(x) dependence (placeholder for now)
# These would typically come from a prior analysis, similar to Twin Primes
FITTED_POLIGNAC_OMEGA_A = 0.0 # Placeholder
FITTED_POLIGNAC_OMEGA_B = 0.0 # Placeholder

def polignac_omega_x_trend_function(omega_x):
    return FITTED_POLIGNAC_OMEGA_A * omega_x + FITTED_POLIGNAC_OMEGA_B

def polignac_count_distinct_prime_factors(n):
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

def get_polignac_double_filtered_residual(data_file):
    print(f"Calculating Polignac double filtered residual from {data_file}...")
    df = pd.read_csv(data_file)
    df_filtered = df[df['x'] > 1].copy()
    
    # Normalization factor based on Hardy-Littlewood conjecture
    # Expected count ~ C_n * x / (log x)^2
    # Error term is (actual - expected)
    # We normalize by sqrt(x) / log(x) to align with other residuals
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    # Apply omega(x) filtering (placeholder for now)
    df_filtered['omega_x'] = df_filtered['x'].apply(polignac_count_distinct_prime_factors)
    df_filtered['predicted_omega_trend'] = df_filtered['omega_x'].apply(polignac_omega_x_trend_function)
    df_filtered['double_filtered_residual'] = df_filtered['normalized_error'] - df_filtered['predicted_omega_trend']
    
    # Return log(x) as time-like variable and the double filtered residual
    print("Polignac residual calculation complete.")
    return np.log(df_filtered['x'].values), df_filtered['double_filtered_residual'].values

if __name__ == "__main__":
    # Example usage:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_file_path = os.path.join(project_root, 'data', 'polignac', 'polignac_data_n4_1M.csv')
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
    else:
        t_polignac, y_polignac = get_polignac_double_filtered_residual(data_file_path)
        print(f"Sample t_polignac (first 5): {t_polignac[:5]}")
        print(f"Sample y_polignac (first 5): {y_polignac[:5]}")
