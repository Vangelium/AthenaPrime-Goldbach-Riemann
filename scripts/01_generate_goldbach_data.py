

import math
import os

def sieve_of_eratosthenes(limit):
    """
    Generates a set of prime numbers up to a given limit using the Sieve of Eratosthenes.
    """
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if primes[i]:
            for multiple in range(i * i, limit + 1, i):
                primes[multiple] = False
    
    prime_numbers = set()
    for i in range(2, limit + 1):
        if primes[i]:
            prime_numbers.add(i)
    return prime_numbers

def generate_goldbach_data(limit, prime_set):
    """
    Generates Goldbach partition counts g(N) for even N up to a given limit.
    """
    goldbach_data = []
    # Iterate through even numbers from 4 to the limit
    for n in range(4, limit + 1, 2):
        count = 0
        # Iterate through primes up to n/2 to find pairs
        for p in prime_set:
            if p > n / 2:
                break
            if (n - p) in prime_set:
                count += 1
        goldbach_data.append((n, count))
        # Progress indicator
        if n % 10000 == 0:
            print(f"Processed up to N = {n}...")
    return goldbach_data

if __name__ == "__main__":
    LIMIT = 1000000
    # Place the output file in the 'data' subdirectory of the project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "goldbach_data_1M.csv")

    print(f"Generating primes up to {LIMIT}...")
    primes = sieve_of_eratosthenes(LIMIT)
    print("Prime generation complete.")

    print(f"Calculating Goldbach partitions g(N) for N up to {LIMIT}...")
    data = generate_goldbach_data(LIMIT, primes)
    print("Calculation complete.")

    print(f"Saving data to {output_file}...")
    try:
        with open(output_file, 'w', newline='') as f:
            f.write("N,g(N)\n")
            for n, count in data:
                f.write(f"{n},{count}\n")
        print(f"Data successfully saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")


