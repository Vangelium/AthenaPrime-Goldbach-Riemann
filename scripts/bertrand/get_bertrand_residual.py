import pandas as pd
import numpy as np
import os

def get_bertrand_residual(data_file):
    """
    Calculates the residual for Bertrand's Postulate.
    The residual is simply the 'error' term from the data generation,
    which is (actual_count - expected_count).
    We normalize it by sqrt(n) / log(n) to align with other residuals.
    """
    print(f"Calculating Bertrand residual from {data_file}...")
    df = pd.read_csv(data_file)
    df_filtered = df[df['n'] > 1].copy() # Ensure n > 1

    # Normalization factor: similar to Twin Primes and Polignac,
    # we use sqrt(x) / log(x) where x is 'n' in this case.
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['n']) / np.log(df_filtered['n'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']
    
    # Return log(n) as time-like variable and the normalized residual
    print("Bertrand residual calculation complete.")
    return np.log(df_filtered['n'].values), df_filtered['normalized_error'].values

if __name__ == "__main__":
    # Example usage:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_file_path = os.path.join(project_root, 'data', 'bertrand', 'bertrand_data_1M.csv')
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
    else:
        t_bertrand, y_bertrand = get_bertrand_residual(data_file_path)
        print(f"Sample t_bertrand (first 5): {t_bertrand[:5]}")
        print(f"Sample y_bertrand (first 5): {y_bertrand[:5]}")
