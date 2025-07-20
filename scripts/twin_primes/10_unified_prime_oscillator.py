import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Constants and Helper Functions (from previous scripts) ---

# Fitted linear function parameters for omega(x) dependence (from 08_model_omega_x_dependence.py)
# y = a * omega_x + b
FITTED_OMEGA_A = 0.4575
FITTED_OMEGA_B = 11.9139

def count_distinct_prime_factors(n):
    """
    Calculates omega(n), the number of distinct prime factors of n.
    """
    if n <= 1:
        return 0
    count = 0
    d = 2
    temp_n = n
    while d * d <= temp_n:
        if temp_n % d == 0:
            count += 1
            while temp_n % d == 0:
                temp_n //= d
        d += 1
    if temp_n > 1:
        count += 1
    return count

# Riemann Zeros (imaginary parts, gamma_k)
# These are the gamma_k values, not gamma_k / (2*pi)
RIEMANN_ZEROS_GAMMA = np.array([
    14.134725141734693790457251983562417282807,
    21.022039638771554992628479596908909000000,
    25.010857580145688765000000000000000000000,
    30.424876125930000000000000000000000000000,
    32.935061587739189759000000000000000000000,
    37.586178158825600000000000000000000000000,
    40.918719012147500000000000000000000000000,
    43.327073540000000000000000000000000000000,
    48.005150881167000000000000000000000000000,
    49.773832477000000000000000000000000000000,
    52.970321477000000000000000000000000000000,
    56.446247698000000000000000000000000000000,
    59.347044000000000000000000000000000000000,
    60.831778525000000000000000000000000000000,
    65.112544000000000000000000000000000000000,
    67.079810000000000000000000000000000000000,
    69.546484000000000000000000000000000000000,
    72.067157000000000000000000000000000000000,
    75.704691000000000000000000000000000000000,
    77.446197000000000000000000000000000000000
])

# --- UnifiedPrimeOscillator Class ---

class UnifiedPrimeOscillator:
    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.riemann_zeros = RIEMANN_ZEROS_GAMMA # Use the global gamma_k values

        # Placeholder parameters for amplitudes and phases
        # In a real scenario, these would be fitted from spectral analysis
        # For demonstration, we'll use some example values or derive from power
        if self.problem_type == 'twin_primes':
            # Example amplitudes (can be scaled from spectral power, e.g., sqrt(power))
            # These are just illustrative, not fitted values
            self.amplitudes = np.sqrt(np.array([
                0.1815, 0.1327, 0.1324, 0.0839, 0.0738, 0.0712, 0.0668, 0.0602, # First 8 significant ones
                0.0403, 0.0347, 0.0335, 0.0335, 0.0335, 0.0335, 0.0335, 0.0335, 0.0335, 0.0335, 0.0335, 0.0335
            ]))
            self.phases = np.linspace(0, 2*np.pi, len(self.riemann_zeros)) # Illustrative phases
            self.omega_a = FITTED_OMEGA_A
            self.omega_b = FITTED_OMEGA_B
        elif self.problem_type == 'goldbach':
            # Placeholder for Goldbach parameters
            self.amplitudes = np.sqrt(np.array([0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])) # Example
            self.phases = np.linspace(0, 2*np.pi, len(self.riemann_zeros)) # Illustrative phases
            # Goldbach structural term parameters (placeholders)
            self.goldbach_C2 = 1.0 # Example
            self.goldbach_alpha = 0.5 # Example
        else:
            raise ValueError("problem_type must be 'twin_primes' or 'goldbach'")

    def structural_term(self, x, omega_x):
        if self.problem_type == 'twin_primes':
            # Based on our fitted omega(x) dependence
            return self.omega_a * omega_x + self.omega_b
        elif self.problem_type == 'goldbach':
            # Placeholder for Goldbach structural term
            # From README: C2 * (-1)**omega * np.sqrt(2**omega / np.log(x))
            # This would need proper implementation of omega(x) for Goldbach context
            # For now, a simplified placeholder
            return self.goldbach_C2 * np.sqrt(omega_x) * np.log(x) # Illustrative

    def riemann_term(self, x):
        log_x = np.log(x)
        # Handle cases where log(x) is not defined or <= 0 (e.g., x <= 1)
        # Set contribution to 0 for such elements
        riemann_contribution = np.zeros_like(x, dtype=float)
        valid_indices = log_x > 0 # Only consider x > 1 for log(x)
        
        if np.any(valid_indices): # Only proceed if there are valid indices
            valid_log_x = log_x[valid_indices]
            
            oscillatory_sum_valid = np.zeros_like(valid_log_x, dtype=float)
            for k in range(len(self.riemann_zeros)):
                oscillatory_sum_valid += self.amplitudes[k] * np.cos(self.riemann_zeros[k] * valid_log_x + self.phases[k])
            
            riemann_contribution[valid_indices] = oscillatory_sum_valid
            
        return riemann_contribution

    def __call__(self, x, omega_x):
        # The model predicts the normalized error
        # For twin primes, it's structural_omega + riemann_oscillations
        # For Goldbach, it would be its own structural + riemann_oscillations
        return self.structural_term(x, omega_x) + self.riemann_term(x)

# --- Cross-Validation Framework (for Twin Primes) ---

def run_cross_validation(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Pre-calculate necessary features
    df_processed = df[df['x'] > 1].copy()
    df_processed['normalization_factor'] = np.sqrt(df_processed['x']) / np.log(df_processed['x'])
    df_processed['normalized_error'] = df_processed['error'] / df_processed['normalization_factor']
    df_processed['omega_x'] = df_processed['x'].apply(count_distinct_prime_factors)

    # Drop rows with NaN/Inf in critical columns
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.dropna(subset=['x', 'normalized_error', 'omega_x'], inplace=True)

    if df_processed.empty:
        print("No valid data points for cross-validation. Exiting.")
        return

    # Split data into training and testing sets (e.g., 50/50)
    # For time-series like data, it's often better to split chronologically
    # Here, we'll do a simple split for demonstration
    train_df, test_df = train_test_split(df_processed, test_size=0.5, random_state=42)

    print(f"Training data size: {len(train_df)}")
    print(f"Testing data size: {len(test_df)}")

    # Initialize the UnifiedPrimeOscillator for twin primes
    model = UnifiedPrimeOscillator(problem_type='twin_primes')

    # --- Prediction and Evaluation ---
    print("Making predictions on test set...")
    # Predict normalized error using the model
    # Note: In a full model, parameters would be fitted on train_df
    # Here, we use pre-fitted omega_x params and illustrative Riemann params
    y_pred_test = model(test_df['x'].values, test_df['omega_x'].values)
    y_true_test = test_df['normalized_error'].values

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true_test, y_pred_test)
    print(f"Mean Squared Error (MSE) on test set: {mse:.6f}")

    # --- Visualization of Predictions vs True Values ---
    os.makedirs(plots_dir, exist_ok=True)

    print("Generating prediction vs true values plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot true values
    ax.scatter(test_df['x'], y_true_test, s=1, alpha=0.5, color='blue', label='True Normalized Error')
    # Plot predicted values
    ax.scatter(test_df['x'], y_pred_test, s=1, alpha=0.5, color='red', label='Predicted Normalized Error')

    ax.set_title('Unified Model Prediction vs True Normalized Error (Twin Primes)', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    ax.legend()
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "10_unified_model_prediction_1M.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")
    plt.close(fig)

    # --- Plotting Residuals (True - Predicted) ---
    print("Generating prediction residuals plot...")
    fig, ax = plt.subplots(figsize=(18, 10))

    residuals = y_true_test - y_pred_test
    ax.scatter(test_df['x'], residuals, s=1, alpha=0.5, color='green')

    ax.set_title('Unified Model Prediction Residuals (Twin Primes)', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Residuals (True - Predicted)', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    plt.tight_layout()
    plot_file_residuals = os.path.join(plots_dir, "11_unified_model_residuals_1M.png")
    plt.savefig(plot_file_residuals, dpi=300)
    print(f"Plot saved to {plot_file_residuals}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    input_data_file = os.path.join(data_dir, "twin_prime_data_1M.csv")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    run_cross_validation(input_data_file, plots_dir)
    print("\nUnified model cross-validation completed.")
