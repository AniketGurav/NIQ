import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Paths
cwd = os.getcwd() + "/genImages/"
CSV_PATH = Path("//cluster/datastore/aniketag/NIQ/cleaned_openfoodfacts.csv")
OUTPUT_FOLDER = Path(cwd)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def basic_distribution_analysis(csv_path: Path, output_folder: Path):
    """
    Plot top 10 value counts for metadata fields:
    - categories, brands, countries, main_category
    """
    df = pd.read_csv(csv_path, low_memory=False)

    key_columns = ['categories', 'brands', 'countries', 'main_category']
    available_columns = [col for col in key_columns if col in df.columns]

    n_cols = min(len(available_columns), 4)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for idx, col in enumerate(available_columns):
        value_counts = df[col].value_counts(dropna=False).head(10)
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[idx])
        axes[idx].set_title(f"Top 10 {col}")
        axes[idx].set_xlabel("Count")
        axes[idx].set_ylabel(col)

    for idx in range(len(available_columns), 4):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plot_path = output_folder / "top_10_metadata_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved metadata plot to {plot_path}")

def label_distribution_analysis(csv_path: Path, output_folder: Path):
    """
    Plot class distribution of:
    - pnns_groups_1 (all)
    - pnns_groups_2 (top 20)
    """
    df = pd.read_csv(csv_path, low_memory=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    if 'pnns_groups_1' in df.columns:
        g1_counts = df['pnns_groups_1'].value_counts()
        sns.barplot(x=g1_counts.values, y=g1_counts.index, ax=axes[0])
        axes[0].set_title("Distribution of pnns_groups_1")
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel("pnns_groups_1")

    if 'pnns_groups_2' in df.columns:
        g2_counts = df['pnns_groups_2'].value_counts().head(20)
        sns.barplot(x=g2_counts.values, y=g2_counts.index, ax=axes[1])
        axes[1].set_title("Top 20 Distribution of pnns_groups_2")
        axes[1].set_xlabel("Count")
        axes[1].set_ylabel("pnns_groups_2")

    plt.tight_layout()
    label_plot_path = output_folder / "label_distributions.png"
    plt.savefig(label_plot_path)
    plt.close()
    print(f"✅ Saved label plot to {label_plot_path}")

def main():
    basic_distribution_analysis(CSV_PATH, OUTPUT_FOLDER)
    label_distribution_analysis(CSV_PATH, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
