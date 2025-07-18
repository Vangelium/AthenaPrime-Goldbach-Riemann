
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_error_term(df, output_path):
    """
    Generates and saves a scatter plot of the error term Delta(N) vs. N.

    Args:
        df (pd.DataFrame): DataFrame containing 'N' and 'Delta(N)' columns.
        output_path (str): The path to save the generated plot image.
    """
    print(f"Generating plot of Delta(N) vs. N...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Using a scatter plot is better to see the distribution without implying continuity
    ax.scatter(df['N'], df['Delta(N)'], s=5, alpha=0.6, edgecolors='none')

    # Professional plot styling
    ax.set_title('Error Term Δ(N) vs. N for Goldbach Partitions (up to N=150k)', fontsize=16)
    ax.set_xlabel('N (Even Numbers)', fontsize=12)
    ax.set_ylabel('Error Term Δ(N) = g(N) - g_est(N)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Format y-axis for better readability if numbers are large
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ',')))
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

    input_file = os.path.join(data_dir, "goldbach_analysis_150k.csv")
    output_plot_file = os.path.join(plots_dir, "01_delta_N_vs_N_150k.png")

    # Check for input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please run '02_calculate_theoretical_estimate.py' first.")
        exit()

    print(f"Reading analysis data from {input_file}...")
    df_analysis = pd.read_csv(input_file)

    # Generate and save the plot
    plot_error_term(df_analysis, output_plot_file)

