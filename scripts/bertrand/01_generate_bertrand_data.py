import pandas as pd
import numpy as np
import os
from sympy import primerange
from scripts.goldbach.goldbach_core_functions import sieve_of_eratosthenes # Reusing prime generation

# --- Constants ---
N_MAX = 1000000  # Maximum 'n' to consider for Bertrand's Postulate
STEP_SIZE = 1000 # Step size for n to generate data points

def generate_bertrand_data(n_max, step_size):
    """
    Generates data for Bertrand's Postulate.
    For each n, it counts primes in (n, 2n) and compares to expected.
    """
    print(f"Generating Bertrand data up to n_max={n_max} with step_size={step_size}...")

    # Generate primes up to 2 * N_MAX for efficient lookup
    all_primes = sieve_of_eratosthenes(2 * n_max)
    all_primes_set = set(all_primes)

    n_values = []
    actual_prime_counts = []
    expected_prime_counts = []

    for n in range(2, n_max + 1, step_size):
        # Count primes in (n, 2n)
        count = 0
        for p in all_primes:
            if p > n and p < 2 * n:
                count += 1
            elif p >= 2 * n:
                break # Primes are sorted, so no need to check further

        n_values.append(n)
        actual_prime_counts.append(count)

        # Expected number of primes in (n, 2n) is approximately (2n/log(2n)) - (n/log(n))
        if n > 1:
            expected = (2 * n / np.log(2 * n)) - (n / np.log(n))
        else:
            expected = 0
        expected_prime_counts.append(expected)

    df = pd.DataFrame({
        'n': n_values,
        'actual_count': actual_prime_counts,
        'expected_count': expected_prime_counts
    })
    df['error'] = df['actual_count'] - df['expected_count']

    print(f"Generated {len(df)} data points.")
    return df

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, 'data', 'bertrand')
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, f"bertrand_data_{N_MAX // 1000000}M.csv")

    bertrand_df = generate_bertrand_data(N_MAX, STEP_SIZE)
    bertrand_df.to_csv(output_file, index=False)
    print(f"Bertrand data saved to {output_file}")
