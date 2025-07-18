

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_modular_pattern(df, modulus, output_path):
    """
    Generates and saves a scatter plot of the normalized error term,
    colored by the residue class of N modulo a given number.

    Args:
        df (pd.DataFrame): DataFrame with N, Delta(N).
        modulus (int): The modulus to use for coloring (e.g., 3, 4, 5).
        output_path (str): The path to save the generated plot image.
    """
    print(f"Generating plot for N mod {modulus}...")

    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])
    df_filtered['N_mod'] = df_filtered['N'] % modulus

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Use a qualitative colormap suitable for discrete categories
    cmap = plt.get_cmap('tab10', modulus)

    scatter = ax.scatter(
        df_filtered['N'], 
        df_filtered['Delta_norm(N)'], 
        c=df_filtered['N_mod'], 
        cmap=cmap, 
        s=8, 
        alpha=0.7
    )

    # Create a legend to identify the residue classes
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  label=f'N mod {modulus} = {i}', 
                                  markerfacecolor=cmap(i), markersize=10) 
                       for i in range(modulus)]
    ax.legend(handles=legend_elements, title=f"Residue Class (mod {modulus})")

    ax.set_title(f'Normalized Error Term Colored by N mod {modulus} (up to N=150k)', fontsize=16)
    ax.set_xlabel('N (Even Numbers)', fontsize=12)
    ax.set_ylabel('Normalized Error Δ(N) / sqrt(N)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=300)
        print(f"Plot successfully saved to {output_path}")
    except IOError as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')

    input_file = os.path.join(data_dir, "goldbach_full_analysis_150k.csv")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit()

    df = pd.read_csv(input_file)

    # List of moduli to analyze
    moduli_to_test = [3, 4, 5, 6]

    for i, mod in enumerate(moduli_to_test):
        output_plot_file = os.path.join(
            plots_dir, f"{i+5:02d}_modular_analysis_mod_{mod}_150k.png"
        )
        plot_modular_pattern(df, mod, output_plot_file)

    print("\nModular analysis plots generated.")

