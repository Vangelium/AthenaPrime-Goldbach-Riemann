import pandas as pd
import numpy as np
import os
import sys
from sympy import primerange
from scripts.goldbach.goldbach_core_functions import sieve_of_eratosthenes # Reusing prime generation

# --- Constants ---
N_MAX = 1000000  # Maximum number to consider for prime pairs
PRIME_GAP_N = 4  # The 'n' in Polignac's Conjecture (e.g., 4 for Cousin Primes)

# Hardy-Littlewood constant C_n for prime k-tuples
# C2 = 0.6601618158... (Twin Prime Constant)
# C_n = 2 * C2 * product_{p|n, p>2} ((p-1)/(p-2))
# For n=4, odd prime factors are none, so product is 1.
# C_4 = 2 * C2 = 2 * 0.6601618158 = 1.3203236316
HARDY_LITTLEWOOD_CN = 1.3203236316 # For n=4

def generate_polignac_data(n_max, prime_gap_n):
    """
    Generates data for Polignac's Conjecture for a given prime gap 'n'.
    Counts the number of prime pairs (p, p+n) up to x, and calculates the
    expected count based on the Hardy-Littlewood conjecture.
    """
    print(f"Generating Polignac data for n={prime_gap_n} up to x={n_max}...")

    primes = sieve_of_eratosthenes(n_max)
    primes_set = set(primes)

    x_values = []
    actual_counts = []
    expected_counts = []

    count = 0
    for i, p in enumerate(primes):
        if p + prime_gap_n in primes_set:
            count += 1
        
        # Record data points at intervals, similar to other data generation
        if i % 1000 == 0 or p == primes[-1]: # Record every 1000th prime or at the end
            x_values.append(p)
            actual_counts.append(count)
            
            # Expected count based on Hardy-Littlewood conjecture
            # pi_n(x) ~ C_n * x / (log x)^2
            if p > 1: # Avoid log(1)
                expected = HARDY_LITTLEWOOD_CN * p / (np.log(p)**2)
            else:
                expected = 0
            expected_counts.append(expected)

    df = pd.DataFrame({
        'x': x_values,
        'actual_count': actual_counts,
        'expected_count': expected_counts
    })
    df['error'] = df['actual_count'] - df['expected_count']

    print(f"Generated {len(df)} data points.")
    return df

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    data_dir = os.path.join(project_root, 'data', 'polignac')
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, f"polignac_data_n{PRIME_GAP_N}_{N_MAX // 1000000}M.csv")

    polignac_df = generate_polignac_data(N_MAX, PRIME_GAP_N)
    polignac_df.to_csv(output_file, index=False)
    print(f"Polignac data saved to {output_file}")
