

import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_twin_prime_error(data_file, plot_file):
    """
    Loads twin prime data and visualizes the error term.
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    print("Generating error term plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.scatter(df['x'], df['error'], s=1, alpha=0.5, color='darkorange')

    ax.set_title('Error Term E₂(x) for Twin Prime Counting Function (up to x=1M)', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Error E₂(x) = π₂(x) - 2C₂x/(ln x)²', fontsize=14)
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
    output_plot_file = os.path.join(plots_dir, "01_twin_prime_error_1M.png")

    if not os.path.exists(input_data_file):
        print(f"Error: Data file not found at {input_data_file}")
        exit()

    visualize_twin_prime_error(input_data_file, output_plot_file)
    print("\nTwin prime error visualization script completed.")

