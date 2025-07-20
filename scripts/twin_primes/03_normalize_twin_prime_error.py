

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def normalize_and_visualize_twin_prime_error(data_file, plot_file):
    """
    Loads twin prime data, normalizes the error term, and visualizes it.
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Calculate the normalization factor: sqrt(x) / log(x)
    # Ensure x > 1 for log(x)
    df_filtered = df[df['x'] > 1].copy()
    df_filtered['normalization_factor'] = np.sqrt(df_filtered['x']) / np.log(df_filtered['x'])
    
    # Calculate the normalized error
    df_filtered['normalized_error'] = df_filtered['error'] / df_filtered['normalization_factor']

    print("Generating normalized error term plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.scatter(df_filtered['x'], df_filtered['normalized_error'], s=1, alpha=0.5, color='darkcyan')

    ax.set_title('Normalized Error Term E₂(x) / (√x / ln x) for Twin Primes (up to x=1M)', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    input_data_file = os.path.join(data_dir, "twin_prime_data_1M.csv")
    output_plot_file = os.path.join(plots_dir, "02_normalized_twin_prime_error_1M.png")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    normalize_and_visualize_twin_prime_error(input_data_file, output_plot_file)
    print("\nNormalized twin prime error visualization script completed.")

