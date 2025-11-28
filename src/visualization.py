import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Utility helpers ----------

def ensure_directories():
    """
    Ensure that all output directories exist.
    """
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)


def load_datasets(raw_path: str, clean_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw and cleaned datasets and normalize column names to lowercase.

    Expected (minimum) columns:
    - raw:   link, short_description, category
    - clean: link, news_text, category

    We DO NOT rely on 'headline' to avoid incorrect assumptions.
    """
    raw = pd.read_csv(raw_path)
    clean = pd.read_csv(clean_path)

    raw.columns = [c.strip().lower() for c in raw.columns]
    clean.columns = [c.strip().lower() for c in clean.columns]

    required_raw = {"link", "short_description", "category"}
    required_clean = {"link", "news_text", "category"}

    missing_raw = required_raw - set(raw.columns)
    missing_clean = required_clean - set(clean.columns)

    if missing_raw:
        raise ValueError(f"Raw dataset is missing required columns: {missing_raw}")
    if missing_clean:
        raise ValueError(f"Cleaned dataset is missing required columns: {missing_clean}")

    return raw, clean


# ---------- Text comparison & metrics ----------

def compare_text_columns(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compare raw short_description with cleaned news_text.

    - Only compares rows that exist in BOTH datasets (inner merge on 'link' + 'category'),
      so we never compare to a non-existing cleaned row.
    - Adds length-based metrics and a simple 'expanded vs short' classification.
    """
    merged = raw.merge(
        clean,
        on=["link", "category"],
        how="inner",
        suffixes=("_raw", "_clean"),
    )

    merged["short_description"] = merged["short_description"].fillna("")
    merged["news_text"] = merged["news_text"].fillna("")

    merged["short_len"] = merged["short_description"].str.len()
    merged["news_len"] = merged["news_text"].str.len()

    # Avoid division by zero
    merged["length_ratio"] = np.where(
        merged["short_len"] > 0,
        merged["news_len"] / merged["short_len"],
        np.nan,
    )

    # Heuristic: consider "expanded" if cleaned text is significantly longer
    merged["is_expanded"] = merged["length_ratio"] > 1.5

    return merged


def compute_dataset_level_stats(raw: pd.DataFrame, clean: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute required high-level metrics:

    - total raw rows
    - total cleaned rows
    - rows with full article text vs short text
    - how many rows were expanded
    - how many rows were lost during cleaning (absolute & percentage)
    """
    total_raw = len(raw)
    total_clean = len(clean)

    raw_links = set(raw["link"])
    clean_links = set(clean["link"])

    lost_links = raw_links - clean_links
    lost_rows_abs = len(lost_links)
    lost_rows_pct = (lost_rows_abs / total_raw * 100) if total_raw > 0 else 0.0

    expanded_rows = merged["is_expanded"].sum()
    short_like_rows = (~merged["is_expanded"]).sum()

    stats = {
        "total_raw_rows": total_raw,
        "total_cleaned_rows": total_clean,
        "rows_in_both_datasets_for_comparison": len(merged),
        "rows_lost_during_cleaning_abs": lost_rows_abs,
        "rows_lost_during_cleaning_pct": lost_rows_pct,
        "rows_with_expanded_full_text": int(expanded_rows),
        "rows_with_short_like_text": int(short_like_rows),
    }

    return pd.DataFrame([stats])


def save_text_comparison_outputs(merged: pd.DataFrame):
    """
    Save detailed comparison outputs to results/metrics:

    - text_comparison_summary.csv : row-by-row comparison
    - text_length_stats.csv       : descriptive stats for lengths & ratios
    - text_comparison_sample.csv  : small sample for manual inspection
    """
    summary_cols = [
        "link",
        "category",
        "short_description",
        "news_text",
        "short_len",
        "news_len",
        "length_ratio",
        "is_expanded",
    ]
    summary = merged[summary_cols].copy()

    summary.to_csv("results/metrics/text_comparison_summary.csv", index=False)

    length_stats = summary[["short_len", "news_len", "length_ratio"]].describe()
    length_stats.to_csv("results/metrics/text_length_stats.csv")

    sample = summary.sample(
        n=min(100, len(summary)), random_state=42
    ) if len(summary) > 0 else summary.head(0)
    sample.to_csv("results/metrics/text_comparison_sample.csv", index=False)


# ---------- Category analysis & plots ----------

def plot_category_distribution(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    output_path: str,
):
    """
    Generic bar plot of category counts.
    Each call writes to a unique file path (provided by caller).
    """
    counts = df[category_col].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def category_analysis(raw: pd.DataFrame, clean: pd.DataFrame, merged: pd.DataFrame):
    """
    Perform category analysis across raw, cleaned, and merged datasets.

    Outputs:
    - results/figures/category_raw.png
    - results/figures/category_cleaned.png
    - results/figures/category_merged.png
    - results/figures/category_comparison_raw_clean_merged.png
    - results/metrics/category_comparison_counts.csv
    """
    # Individual distributions
    plot_category_distribution(
        raw,
        category_col="category",
        title="Category Distribution - RAW dataset",
        output_path="results/figures/category_raw.png",
    )

    plot_category_distribution(
        clean,
        category_col="category",
        title="Category Distribution - CLEANED dataset",
        output_path="results/figures/category_cleaned.png",
    )

    plot_category_distribution(
        merged,
        category_col="category",
        title="Category Distribution - MERGED (rows in both)",
        output_path="results/figures/category_merged.png",
    )

    # Combined comparison table
    raw_counts = raw["category"].value_counts().rename("raw_count")
    clean_counts = clean["category"].value_counts().rename("clean_count")
    merged_counts = merged["category"].value_counts().rename("merged_count")

    all_categories = sorted(
        set(raw_counts.index) |
        set(clean_counts.index) |
        set(merged_counts.index)
    )

    comp_df = pd.DataFrame(index=all_categories)
    comp_df["raw_count"] = raw_counts
    comp_df["clean_count"] = clean_counts
    comp_df["merged_count"] = merged_counts
    comp_df = comp_df.fillna(0).astype(int)

    comp_df.to_csv("results/metrics/category_comparison_counts.csv")

    # Grouped bar chart
    x = np.arange(len(all_categories))
    width = 0.25

    plt.figure(figsize=(12, 7))
    plt.bar(x - width, comp_df["raw_count"], width, label="Raw")
    plt.bar(x, comp_df["clean_count"], width, label="Cleaned")
    plt.bar(x + width, comp_df["merged_count"], width, label="Merged (inner)")

    plt.xticks(x, all_categories, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Category Distribution Comparison: Raw vs Cleaned vs Merged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/category_comparison_raw_clean_merged.png")
    plt.close()


# ---------- Main entrypoint ----------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualization & analysis script.\n"
            "Compares raw short_description with cleaned news_text,\n"
            "computes required metrics, and generates category plots."
        )
    )
    parser.add_argument(
        "--raw_path",
        default="Data/raw/raw.csv",
        help="Path to RAW dataset CSV (default: Data/raw/raw.csv)",
    )
    parser.add_argument(
        "--clean_path",
        default="Data/processed/processed.csv",
        help="Path to CLEANED dataset CSV (default: Data/processed/processed.csv)",
    )

    args = parser.parse_args()

    ensure_directories()

    print(f"Loading datasets:\n  RAW   = {args.raw_path}\n  CLEAN = {args.clean_path}")
    raw, clean = load_datasets(args.raw_path, args.clean_path)

    print("Comparing short_description (raw) with news_text (cleaned)...")
    merged = compare_text_columns(raw, clean)

    print("Computing dataset-level statistics...")
    stats_df = compute_dataset_level_stats(raw, clean, merged)
    stats_df.to_csv("results/metrics/dataset_level_stats.csv", index=False)

    print("Saving detailed text comparison outputs...")
    save_text_comparison_outputs(merged)

    print("Running category analysis...")
    category_analysis(raw, clean, merged)

    print("All visualizations and metrics have been generated in 'results/figures' and 'results/metrics'.")


if __name__ == "__main__":
    main()
