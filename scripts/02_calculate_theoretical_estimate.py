
import pandas as pd
import numpy as np
import os

def prime_factors(n):
    """Returns the set of distinct prime factors of a given integer."""
    factors = set()
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
    return factors

def singular_series(n, twin_prime_constant):
    """Calculates the Singular Series S(N) for a given even number N."""
    if n % 2 != 0:
        return 0
    
    factors = prime_factors(n)
    product = 1.0
    for p in factors:
        if p != 2:
            product *= (p - 1) / (p - 2)
            
    return twin_prime_constant * product

def li2_approx(x):
    """Approximation for the Logarithmic Integral Li2(x)."""
    return x / (np.log(x)**2)

if __name__ == "__main__":
    # Constants
    TWIN_PRIME_CONSTANT = 0.6601618158468695739278121100145557784326

    # File paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    input_file = os.path.join(data_dir, "goldbach_data_1M.csv")
    output_file = os.path.join(data_dir, "goldbach_analysis_1M.csv")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please run '01_generate_goldbach_data.py' first.")
        exit()

    print(f"Reading Goldbach data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Calculating theoretical estimates...")
    # Calculate Singular Series S(N)
    df['S(N)'] = df['N'].apply(lambda n: singular_series(n, TWIN_PRIME_CONSTANT))

    # Calculate Logarithmic Integral Li2(N)
    # We use an approximation, for N > 4 to avoid log(0) or log(1)
    df['Li2(N)'] = df['N'].apply(lambda n: li2_approx(n) if n > 1 else 0)

    # Calculate the theoretical estimate g_est(N)
    df['g_est(N)'] = df['S(N)'] * df['Li2(N)']

    # Calculate the error term Delta(N)
    df['Delta(N)'] = df['g(N)'] - df['g_est(N)']
    print("Calculations complete.")

    print(f"Saving analysis to {output_file}...")
    try:
        df.to_csv(output_file, index=False, float_format='%.8f')
        print(f"Analysis successfully saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")

