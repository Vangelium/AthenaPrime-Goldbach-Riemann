
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_distinct_prime_factors(n):
    """Returns the number of distinct prime factors of n."""
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
    return len(factors)

def plot_error_by_omega(df, output_path):
    """
    Generates and saves a scatter plot of the normalized error term,
    colored by the number of distinct prime factors (omega(N)).

    Args:
        df (pd.DataFrame): DataFrame containing 'N', 'Delta(N)', and 'omega(N)'.
        output_path (str): The path to save the generated plot image.
    """
    print("Generating plot of normalized Delta(N) colored by omega(N)...")

    df_filtered = df[df['N'] > 0].copy()
    df_filtered['Delta_norm(N)'] = df_filtered['Delta(N)'] / np.sqrt(df_filtered['N'])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Create a scatter plot with colors mapped to omega(N)
    scatter = ax.scatter(
        df_filtered['N'], 
        df_filtered['Delta_norm(N)'], 
        c=df_filtered['omega(N)'], 
        cmap='viridis', # A perceptually uniform colormap
        s=8, 
        alpha=0.7
    )

    # Add a colorbar to interpret the colors
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ω(N) - Number of Distinct Prime Factors', fontsize=12)

    # Professional plot styling
    ax.set_title('Normalized Error Term Δ(N)/sqrt(N) vs. N, Colored by ω(N)', fontsize=16)
    ax.set_xlabel('N (Even Numbers)', fontsize=12)
    ax.set_ylabel('Normalized Error Δ(N) / sqrt(N)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300)
        print(f"Plot successfully saved to {output_path}")
    except IOError as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)

if __name__ == "__main__":
    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    input_file = os.path.join(data_dir, "goldbach_analysis_1M.csv")
    output_data_file = os.path.join(data_dir, "goldbach_full_analysis_1M.csv")
    output_plot_file = os.path.join(plots_dir, "03_delta_N_vs_N_colored_by_omega_150k.png")

    # Check for input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit()

    # In this scaled-up version, we focus only on data generation, not plotting.
    print(f"Reading analysis data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Calculating omega(N) for each N up to 1,000,000... (This will take a significant amount of time)")
    df['omega(N)'] = df['N'].apply(get_distinct_prime_factors)
    print("omega(N) calculation complete.")

    # Save the new dataframe with the omega(N) column
    print(f"Saving extended analysis data to {output_data_file}...")
    try:
        df.to_csv(output_data_file, index=False, float_format='%.8f')
        print(f"Data successfully saved to {output_data_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")
