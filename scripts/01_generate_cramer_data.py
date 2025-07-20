
import math
import os
import pandas as pd

def sieve_of_eratosthenes(limit):
    """
    Generates a list of prime numbers up to a given limit using the Sieve of Eratosthenes.
    """
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if primes[i]:
            for multiple in range(i * i, limit + 1, i):
                primes[multiple] = False
    return [i for i, is_prime in enumerate(primes) if is_prime]

def generate_cramer_data(limit):
    """
    Generates data for Cramér's Conjecture: primes, gaps, and error terms.
    """
    print(f"Generating primes up to {limit}...")
    prime_list = sieve_of_eratosthenes(limit)
    print(f"Found {len(prime_list)} primes.")

    data = []
    # Start from the second prime to calculate gaps
    for i in range(len(prime_list) - 1):
        p_n = prime_list[i]
        p_n_plus_1 = prime_list[i+1]
        g_n = p_n_plus_1 - p_n

        # Calculate theoretical gap according to Cramér's conjecture
        # Use max(1, ...) to avoid log(1) issues for very small primes, though primes start from 2
        log_p_n = math.log(p_n)
        theoretical_gap = log_p_n**2

        # Calculate the raw error term
        error_term = g_n - theoretical_gap

        data.append({
            'p_n': p_n,
            'p_n_plus_1': p_n_plus_1,
            'g_n': g_n,
            'log_p_n': log_p_n,
            'theoretical_gap': theoretical_gap,
            'error_term': error_term
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    LIMIT = 1000000 # Initial limit for data generation
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Ensure paths are relative to the new project root for AthenaCramer
    # The '..' correctly points to AthenaCramer/
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"cramer_data_{LIMIT}.csv")

    print(f"Generating Cramér's conjecture data up to x = {LIMIT}...")
    df_cramer = generate_cramer_data(LIMIT)
    print("Cramér's data generation complete.")

    print(f"Saving data to {output_file}...")
    try:
        df_cramer.to_csv(output_file, index=False, float_format='%.8f')
        print(f"Data successfully saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")
