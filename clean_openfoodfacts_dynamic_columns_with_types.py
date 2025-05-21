import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Paths
cwd = os.getcwd() + "//"
CSV_PATH = Path("//cluster/datastore/aniketag/NIQ/cleaned_openfoodfacts.csv")
OUTPUT_FOLDER = Path(cwd)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def basic_distribution_analysis(csv_path: Path, output_folder: Path):
    """
    Compute value counts for key columns (categories, brands, countries, main_category)
    and plot top 10 values in a single 2x2 grid image.
    
    Args:
        csv_path (Path): Path to cleaned CSV.
        output_folder (Path): Path to save the combined plot.
    """
    # Read the cleaned CSV
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Key columns for analysis
    key_columns = ['categories', 'brands', 'countries', 'main_category']
    available_columns = [col for col in key_columns if col in df.columns]
    
    if not available_columns:
        print("None of the key columns found in the dataset")
        return
    
    # Set up a 2x2 grid for plotting
    n_cols = min(len(available_columns), 4)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    # Compute and plot value counts
    print("\n=== Value Counts for Key Columns ===")
    for idx, col in enumerate(available_columns):
        print(f"\nValue Counts for {col}:")
        value_counts = df[col].value_counts(dropna=False).head(10)
        print(value_counts)
        
        # Plot top 10 values as bar plot
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[idx])
        axes[idx].set_title(f"Top 10 {col}")
        axes[idx].set_xlabel("Count")
        axes[idx].set_ylabel(col)
    
    # Remove empty subplots if fewer than 4 columns
    for idx in range(len(available_columns), 4):
        fig.delaxes(axes[idx])
    
    # Save the combined plot
    plt.tight_layout()
    plot_path = output_folder / "top_10_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved combined plot to {plot_path}")

def main():
    basic_distribution_analysis(CSV_PATH, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()