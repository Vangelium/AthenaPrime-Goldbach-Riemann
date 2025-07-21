import pandas as pd
import numpy as np
import os
from scripts.goldbach.goldbach_core_functions import sieve_of_eratosthenes

# --- Constants ---
N_MAX = 1000000  # Maximum even number to consider for Chen's Theorem

# Hardy-Littlewood constant for Goldbach (used as a base for Chen's approximation)
# C2 = 0.6601618158... (Twin Prime Constant)
# C_Goldbach = 2 * C2 = 1.3203236316
# Chen's approximation is more complex, but often related to Goldbach's constant
# For simplicity, we'll use a modified Goldbach-like constant for expected value
CHEN_CONSTANT_APPROX = 0.78 # A rough approximation for the constant in Chen's theorem

def is_semiprime(n, primes_set):
    """Checks if a number is a semiprime (product of two primes)."""
    if n < 4:
        return False
    for p in primes_set:
        if p * p > n:
            break
        if n % p == 0:
            if (n // p) in primes_set:
                return True
    return False

def generate_chen_data(n_max):
    """
    Generates data for Chen's Theorem.
    Counts representations of even numbers as p + q, where q is prime or semiprime.
    """
    print(f"Generating Chen data up to n_max={n_max}...")

    primes = sieve_of_eratosthenes(n_max)
    primes_set = set(primes)

    even_numbers = np.arange(4, n_max + 1, 2) # Only even numbers >= 4

    n_values = []
    actual_counts = []
    expected_counts = []

    for n in even_numbers:
        count = 0
        for p in primes:
            if p >= n:
                break
            q = n - p
            if q in primes_set or is_semiprime(q, primes_set):
                count += 1
        
        n_values.append(n)
        actual_counts.append(count)

        # Expected count based on a simplified Chen-like approximation
        # This is a very rough approximation for demonstration purposes
        if n > 1:
            expected = CHEN_CONSTANT_APPROX * n / (np.log(n)**2)
        else:
            expected = 0
        expected_counts.append(expected)

    df = pd.DataFrame({
        'n': n_values,
        'actual_count': actual_counts,
        'expected_count': expected_counts
    })
    df['error'] = df['actual_count'] - df['expected_count']

    print(f"Generated {len(df)} data points.")
    return df

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, 'data', 'chen')
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, f"chen_data_{N_MAX // 1000000}M.csv")

    chen_df = generate_chen_data(N_MAX)
    chen_df.to_csv(output_file, index=False)
    print(f"Chen data saved to {output_file}")
