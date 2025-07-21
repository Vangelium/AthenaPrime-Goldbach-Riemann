
import numpy as np
from scipy.signal import butter, filtfilt
from scripts.goldbach.goldbach_core_functions import compute_goldbach_sequence, goldbach_asymptotic

def goldbach_residual_clean(n_max):
    """
    Calculates the clean residual of Goldbach.
    Follows the same process as Cramér and Twin Primes.
    """
    print(f"Calculating Goldbach clean residual up to n_max = {n_max}...")

    # 1. Calculate actual Goldbach sequence G(n)
    goldbach_actual_values, even_numbers = compute_goldbach_sequence(n_max)

    # 2. Calculate asymptotic function values
    goldbach_asymptotic_vals = np.array([
        goldbach_asymptotic(n) for n in even_numbers
    ])

    # 3. Calculate raw residual
    raw_residual = goldbach_actual_values - goldbach_asymptotic_vals

    print(f"  Raw Residual - Mean: {np.mean(raw_residual):.4f}, Std: {np.std(raw_residual):.4f}")

    # 4. Apply cleaning filters (high-pass filter to remove long-term trends)
    # For now, we skip the high-pass filter to see the raw spectral content.
    filtered_residual = raw_residual

    # 5. Normalization (mean-centering and scaling to unit variance)
    # Avoid division by zero if std is 0
    std_dev = np.std(filtered_residual)
    if std_dev == 0:
        clean_residual = filtered_residual - np.mean(filtered_residual)
    else:
        clean_residual = (filtered_residual - np.mean(filtered_residual)) / std_dev
    print(f"  Clean Residual - Mean: {np.mean(clean_residual):.4f}, Std: {np.std(clean_residual):.4f}")

    print("Goldbach clean residual calculation complete.")
    return clean_residual, even_numbers
