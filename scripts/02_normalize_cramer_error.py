

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def normalize_cramer_error(data_file, plots_dir):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Calculate the normalized error term
    # Ensure log_p_n is not zero or too small to avoid division by zero/large numbers
    df['normalized_error'] = df['error_term'] / df['log_p_n']

    os.makedirs(plots_dir, exist_ok=True)

    # Plotting the normalized error
    print("Generating normalized error plot for Cramér's Conjecture...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.scatter(df['p_n'], df['normalized_error'], s=1, alpha=0.5, color='darkblue')

    ax.set_title('Normalized Error for Cramér\'s Conjecture: (g_n - (ln p_n)^2) / ln p_n', fontsize=18)
    ax.set_xlabel('p_n (Prime Number)', fontsize=14)
    ax.set_ylabel('Normalized Error', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x_val, p: format(int(x_val), ',')))
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "02_normalized_cramer_error_1M.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Ensure paths are relative to the new project root for AthenaCramer
    # The '..' correctly points to AthenaCramer/
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    input_data_file = os.path.join(data_dir, "cramer_data_1000000.csv")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    normalize_cramer_error(input_data_file, plots_dir)
    print("\nNormalized Cramér error visualization completed.")

