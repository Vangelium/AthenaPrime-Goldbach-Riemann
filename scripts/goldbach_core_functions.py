
import math
import numpy as np

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

def goldbach_partition_count(n, prime_set):
    """
    Counts representations of n as sum of two primes.
    Optimized using set lookup O(sqrt(n)).
    """
    if n % 2 != 0 or n < 4:
        return 0
        
    count = 0
    # Iterate through primes up to n // 2 to avoid double counting
    for p in prime_set:
        if p > n // 2:
            break
        if (n - p) in prime_set:
            count += 1
    return count

def compute_goldbach_sequence(n_max):
    """
    Calculates G(n) for even numbers up to n_max.
    """
    print(f"Generating primes up to {n_max} for Goldbach sequence...")
    primes = sieve_of_eratosthenes(n_max)
    prime_set = set(primes)
    print(f"Found {len(primes)} primes for Goldbach.")

    goldbach_values = []
    even_numbers = np.arange(4, n_max + 1, 2) # Only even numbers >= 4
    
    print(f"Computing Goldbach partition counts for {len(even_numbers)} even numbers...")
    for n in even_numbers:
        goldbach_values.append(goldbach_partition_count(n, prime_set))
    
    return np.array(goldbach_values), even_numbers

def goldbach_asymptotic(n):
    """
    Hardy-Littlewood asymptotic approximation for G(n).
    """
    if n % 2 != 0 or n < 4:
        return 0
        
    C = 1.32032363169373914785562422  # Twin prime constant
        
    # Main term
    main_term = C * n / (np.log(n) ** 2)
        
    # Correction factor for prime divisors of n
    correction_factor = 1.0
    temp_n = n
    p = 3  # Start with p=3 (p=2 does not contribute for even n)
        
    while p * p <= temp_n:
        if temp_n % p == 0:
            correction_factor *= (p - 1) / (p - 2)
            while temp_n % p == 0:
                temp_n //= p
        p += 2  # Only odd primes
        
    if temp_n > 1:  # temp_n is a prime
        # If temp_n is a prime factor of n, apply correction
        # This handles the case where n is a prime itself (not applicable for even n)
        # or if the remaining temp_n is a prime factor
        if temp_n % 2 != 0: # Ensure it's an odd prime factor
            correction_factor *= (temp_n - 1) / (temp_n - 2)
            
    return main_term * correction_factor

