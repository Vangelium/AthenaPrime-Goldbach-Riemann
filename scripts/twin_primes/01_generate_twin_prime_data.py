
import math
import os
import pandas as pd

TWIN_PRIME_CONSTANT_C2 = 0.6601618158

def sieve_of_eratosthenes(limit):
    """
    Generates a boolean array where primes[i] is True if i is prime.
    """
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if primes[i]:
            for multiple in range(i * i, limit + 1, i):
                primes[multiple] = False
    return primes

def generate_twin_prime_data(limit):
    """
    Generates twin prime counts pi_2(x) and theoretical estimates.
    """
    is_prime = sieve_of_eratosthenes(limit)
    
    data = []
    twin_prime_count = 0
    
    # Iterate through numbers to find twin primes
    for x in range(3, limit + 1):
        if is_prime[x] and (x + 2 <= limit and is_prime[x + 2]):
            twin_prime_count += 1
        
        # Calculate theoretical estimate for current x
        if x >= 3: # ln(x) is defined
            theoretical_pi2_x = 2 * TWIN_PRIME_CONSTANT_C2 * (x / (math.log(x)**2))
            error = twin_prime_count - theoretical_pi2_x
            data.append({'x': x, 'pi_2(x)': twin_prime_count, 'theoretical_pi_2(x)': theoretical_pi2_x, 'error': error})
        
        if x % 100000 == 0:
            print(f"Processed up to x = {x}...")
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    LIMIT = 1000000
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "twin_prime_data_1M.csv")

    print(f"Generating twin prime data up to x = {LIMIT}...")
    df_twin_primes = generate_twin_prime_data(LIMIT)
    print("Twin prime data generation complete.")

    print(f"Saving data to {output_file}...")
    try:
        df_twin_primes.to_csv(output_file, index=False, float_format='%.8f')
        print(f"Data successfully saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")
