import pandas as pd
import numpy as np
import os

def get_chen_residual(data_file):
    """
    Calculates the residual for Chen's Theorem.
    The residual is simply the 'error' term from the data generation,
    which is (actual_count - expected_count).
    We normalize it by sqrt(n) / log(n) to align with other residuals.
    """
    print(f"Calculating Chen residual from {data_file}...")
    df = pd.read_csv(data_file)
    df_filtered = df[df['n'] > 1].copy() # Ensure n > 1

    # Normalization factor: similar to Goldbach, Twin Primes, Polignac, Bertrand
    # We use sqrt(n) / log(n) where n is the even number
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['n']) / np.log(df_filtered['n'])
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']
    
    # Return log(n) as time-like variable and the normalized residual
    print("Chen residual calculation complete.")
    return np.log(df_filtered['n'].values), df_filtered['normalized_error'].values

if __name__ == "__main__":
    # Example usage:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_file_path = os.path.join(project_root, 'data', 'chen', 'chen_data_1M.csv')
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
    else:
        t_chen, y_chen = get_chen_residual(data_file_path)
        print(f"Sample t_chen (first 5): {t_chen[:5]}")
        print(f"Sample y_chen (first 5): {y_chen[:5]}")
