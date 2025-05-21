import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.impute import SimpleImputer
import time

# Paths
cwd = os.getcwd() + "/"
CSV_PATH = Path("/cluster/datastore/aniketag/NIQ/cleaned_openfoodfacts.csv")
OUTPUT_FOLDER = Path(cwd)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def summarize_columns(df, title: str, output_folder: Path, tag: str):
    print(f"\n=== {title} ===")
    print(f"Total columns: {len(df.columns)}")
    summary_data = []
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        missing_pct = (n_missing / len(df)) * 100
        summary_data.append((col, dtype, n_unique, n_missing, missing_pct))
    summary_df = pd.DataFrame(summary_data, columns=['Column', 'Dtype', 'Unique', 'Missing', 'Missing %'])
    print(summary_df)

    # Save summary to CSV
    summary_filename = output_folder / f"summary_{tag}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"📄 Saved {title} summary to: {summary_filename}")
    return summary_df

def impute_data(csv_path: Path, output_folder: Path,
                numeric_strategy: str = 'median', categorical_strategy: str = 'most_frequent',
                numeric_fill_value: float = 0.0, categorical_fill_value: str = 'Unknown'):
    print(f"\n📂 Input CSV: {csv_path}")
    start_time = time.time()

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded data with shape: {df.shape}")

    # Summarize before imputation
    summarize_columns(df, title="Before Imputation", output_folder=output_folder, tag="before_imputation")

    # Detect numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Heuristic: Add low-cardinality (<50 unique) as categorical if not numeric
    low_card_cols = [col for col in df.columns
                     if df[col].nunique() < 50 and col not in numeric_cols and col not in categorical_cols]
    categorical_cols = list(set(categorical_cols + low_card_cols))

    print(f"\n🔍 Found {len(numeric_cols)} numeric columns.")
    print(f"🔍 Found {len(categorical_cols)} categorical columns (including heuristic).")
    if low_card_cols:
        print(f"🧐 Added {len(low_card_cols)} low-cardinality columns as categorical: {low_card_cols}")

    # Impute numeric columns
    if numeric_cols:
        t0 = time.time()
        num_imputer = SimpleImputer(strategy=numeric_strategy, fill_value=numeric_fill_value)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        print(f"✅ Numeric imputation complete with strategy='{numeric_strategy}' for {len(numeric_cols)} columns in {time.time() - t0:.2f} sec.")

    # Impute categorical columns
    if categorical_cols:
        t0 = time.time()
        cat_imputer = SimpleImputer(strategy=categorical_strategy, fill_value=categorical_fill_value)
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        print(f"✅ Categorical imputation complete with strategy='{categorical_strategy}' for {len(categorical_cols)} columns in {time.time() - t0:.2f} sec.")

    # Check missing values
    total_missing = df.isna().sum().sum()
    print(f"\n✅ Total missing values after imputation: {total_missing}")

    # Summarize after imputation
    summarize_columns(df, title="After Imputation", output_folder=output_folder, tag="after_imputation")

    # Save to CSV with clear filename
    out_filename = output_folder / f"imputed_openfoodfacts_{numeric_strategy}_{categorical_strategy}_.csv"
    df.to_csv(out_filename, index=False)
    print(f"\n💾 Saved imputed dataset to: {out_filename}")
    print(f"⏱️ Completed in {time.time() - start_time:.2f} seconds.")

def main():
    # Median for numeric, most frequent for categorical
    impute_data(
        CSV_PATH,
        OUTPUT_FOLDER,
        numeric_strategy='median',
        categorical_strategy='most_frequent'
    )

if __name__ == "__main__":
    main()
