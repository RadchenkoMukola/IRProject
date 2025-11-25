from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------

# If this file is inside `src/`, PROJECT_ROOT will be the repo root.
# If you move this file, adjust parents[...] index.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(".").resolve()

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_PATH = DATA_DIR / "raw" / "News_Category_Dataset_v3.csv"
CLEANED_DATA_PATH = DATA_DIR / "processed" / "News_Category_Dataset_v3_cleaned.csv"
MERGED_DATA_PATH = DATA_DIR / "processed" / "News_Category_Dataset_v3_merged.csv"

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# Load datasets
# --------------------------------------------------------------------

def load_datasets():
    """Load the raw, cleaned and merged datasets."""
    print("Loading datasets...")
    raw = pd.read_csv(RAW_DATA_PATH)
    cleaned = pd.read_csv(CLEANED_DATA_PATH)
    merged = pd.read_csv(MERGED_DATA_PATH)

    print(f"Raw rows:     {len(raw)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Merged rows:  {len(merged)}")

    return raw, cleaned, merged


# --------------------------------------------------------------------
# Category distribution
# --------------------------------------------------------------------

def plot_category_distribution(df, title, filename, category_col="category"):
    if category_col not in df.columns:
        print(f"[WARN] '{category_col}' not found in dataframe.")
        return

    counts = df[category_col].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of texts")
    plt.tight_layout()

    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_category_distribution_merged(merged_df):
    if "category" not in merged_df.columns or "category_merged" not in merged_df.columns:
        print("[WARN] Missing category columns in merged dataset.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    merged_df["category"].value_counts().sort_values(ascending=False).plot(
        kind="bar", ax=axes[0]
    )
    axes[0].set_title("Original categories")
    axes[0].set_xlabel("Category")

    merged_df["category_merged"].value_counts().sort_values(ascending=False).plot(
        kind="bar", ax=axes[1]
    )
    axes[1].set_title("Merged categories")
    axes[1].set_xlabel("Merged category")

    plt.tight_layout()
    out_path = FIGURES_DIR / "category_distribution_original_vs_merged.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# --------------------------------------------------------------------
# Row counts before/after processing
# --------------------------------------------------------------------

def plot_row_counts(raw_df, cleaned_df, merged_df):
    stages = ["Raw", "Cleaned", "Merged"]
    counts = [len(raw_df), len(cleaned_df), len(merged_df)]

    plt.figure(figsize=(6, 4))
    plt.bar(stages, counts)
    plt.title("Number of rows at each processing stage")
    plt.ylabel("Number of rows")

    for i, c in enumerate(counts):
        plt.text(i, c, str(c), ha="center", va="bottom")

    plt.tight_layout()
    out_path = FIGURES_DIR / "row_counts_per_stage.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Rows lost raw → cleaned: {counts[0] - counts[1]}")
    print(f"Rows lost cleaned → merged: {counts[1] - counts[2]}")


# --------------------------------------------------------------------
# Text comparison analysis
# --------------------------------------------------------------------

def analyze_text_changes(raw_df, cleaned_df):
    print("Comparing short_description vs news_text...")

    if "link" not in raw_df or "short_description" not in raw_df:
        raise KeyError("Raw dataset missing required columns.")
    if "link" not in cleaned_df or "news_text" not in cleaned_df:
        raise KeyError("Cleaned dataset missing required columns.")

    merged = raw_df[["link", "short_description"]].merge(
        cleaned_df[["link", "news_text"]],
        on="link",
        how="inner",
    )

    print(f"Matched rows: {len(merged)}")

    # same vs different text
    same_mask = merged["short_description"].fillna("") == merged["news_text"].fillna("")
    num_same = same_mask.sum()
    num_diff = (~same_mask).sum()

    print(f"Same text: {num_same}")
    print(f"Different text: {num_diff}")

    # length ratio
    merged["short_len"] = merged["short_description"].fillna("").str.len()
    merged["news_len"] = merged["news_text"].fillna("").str.len()
    merged["length_ratio"] = np.where(
        merged["short_len"] > 0,
        merged["news_len"] / merged["short_len"],
        np.nan,
    )

    # plot: same vs different
    plt.figure(figsize=(5, 4))
    plt.bar(["Same", "Different"], [num_same, num_diff])
    plt.title("Exact text match")
    plt.ylabel("Rows")
    plt.tight_layout()
    out_path = FIGURES_DIR / "text_same_vs_different.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # histogram ratio
    ratios = merged["length_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7, 4))
    plt.hist(ratios, bins=50)
    plt.title("Length ratio (news_text / short_description)")
    plt.xlabel("Ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = FIGURES_DIR / "text_length_ratio_histogram.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # substring check
    print("Checking substring containment...")
    substring_mask = merged.apply(
        lambda row: str(row["short_description"]) in str(row["news_text"]),
        axis=1,
    )
    num_contains = substring_mask.sum()

    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Contains", "Does not contain"],
        [num_contains, len(merged) - num_contains],
    )
    plt.title("Full text contains the original short description?")
    plt.tight_layout()
    out_path = FIGURES_DIR / "text_contains_short_description.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    raw_df, cleaned_df, merged_df = load_datasets()

    plot_category_distribution(
        raw_df,
        "Category distribution (raw)",
        "category_distribution_raw.png"
    )

    plot_category_distribution(
        cleaned_df,
        "Category distribution (cleaned)",
        "category_distribution_cleaned.png"
    )

    plot_category_distribution_merged(merged_df)

    plot_row_counts(raw_df, cleaned_df, merged_df)

    analyze_text_changes(raw_df, cleaned_df)


if __name__ == "__main__":
    main()

