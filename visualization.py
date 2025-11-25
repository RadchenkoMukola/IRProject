# visualization.py
"""
visualization IRProject.

"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

#Helpers
def ensure_out_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {p}")
        return None
    try:
        df = pd.read_csv(p)
        print(f"[LOADED] {p.name} -> shape={df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Could not read {p}: {e}")
        return None

def similarity(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a), str(b)).ratio()

#Paths
project_root = Path(__file__).resolve().parent
raw_path = project_root / "Data" / "raw" / "News_Category_Dataset_v3.csv"
cleaned_path = project_root / "Data" / "processed" / "News_Category_Dataset_v3_cleaned.csv"
merged_path = project_root / "Data" / "processed" / "News_Category_Dataset_v3_merged.csv"
out_dir = ensure_out_dir(project_root / "results" / "figures")

#Load datasets
raw = safe_read_csv(raw_path)
cleaned = safe_read_csv(cleaned_path)
merged = safe_read_csv(merged_path)

# Print columns for debugging
if raw is not None:
    print("RAW columns:", raw.columns.tolist())
if cleaned is not None:
    print("CLEANED columns:", cleaned.columns.tolist())
if merged is not None:
    print("MERGED columns:", merged.columns.tolist())

# Plot 1: Number of rows in each dataset
def plot_row_counts(raw_df, cleaned_df, merged_df, out_p):
    labels = []
    values = []
    if raw_df is not None:
        labels.append("Raw"); values.append(len(raw_df))
    if cleaned_df is not None:
        labels.append("Cleaned"); values.append(len(cleaned_df))
    if merged_df is not None:
        labels.append("Merged"); values.append(len(merged_df))
    if not labels:
        print("[WARN] No dataframes available for row counts.")
        return
    plt.figure(figsize=(7,5))
    plt.bar(labels, values)
    plt.title("Number of rows (Raw, Cleaned, Merged)")
    plt.ylabel("Rows")
    for i, v in enumerate(values):
        plt.text(i, v + max(values)*0.01, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()
    print(f"[SAVED] {out_p}")

plot_row_counts(raw, cleaned, merged, out_dir / "row_counts.png")

#Plot 2: Category counts
def plot_category_counts(dfs, out_p):
    for df in dfs:
        if df is None:
            continue
        # prefer 'category' column
        for col in ['category', 'category_merged', 'news_desk', 'section']:
            if col in df.columns:
                counts = df[col].value_counts().head(20)
                plt.figure(figsize=(12,5))
                counts.plot(kind='bar')
                plt.title(f"Top categories (column: {col})")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(out_p)
                plt.close()
                print(f"[SAVED] {out_p} (from column '{col}')")
                return
    print("[WARN] No category column found in provided dataframes.")

plot_category_counts([raw, cleaned, merged], out_dir / "category_counts.png")

#Plot 3: Text length before vs after cleaning
def plot_length_distribution(raw_df, cleaned_df, out_p):
    raw_text_col_candidates = ['short_description', 'short_desc', 'summary', 'headline', 'text']
    cleaned_text_col_candidates = ['news_text', 'cleaned_text', 'article_text', 'text']
    plotted = False
    plt.figure(figsize=(8,6))
    if raw_df is not None:
        raw_col = next((c for c in raw_text_col_candidates if c in raw_df.columns), None)
        if raw_col:
            raw_lens = raw_df[raw_col].fillna("").astype(str).map(len)
            plt.hist(raw_lens, bins=50, alpha=0.5, label=f"Raw ({raw_col})")
            plotted = True
    if cleaned_df is not None:
        cleaned_col = next((c for c in cleaned_text_col_candidates if c in cleaned_df.columns), None)
        if cleaned_col:
            cleaned_lens = cleaned_df[cleaned_col].fillna("").astype(str).map(len)
            plt.hist(cleaned_lens, bins=50, alpha=0.5, label=f"Cleaned ({cleaned_col})")
            plotted = True
    if not plotted:
        print("[WARN] Could not find suitable text columns for length distribution.")
        return
    plt.legend()
    plt.xlabel("Text length (chars)")
    plt.ylabel("Count")
    plt.title("Text length distribution: raw vs cleaned")
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()
    print(f"[SAVED] {out_p}")

plot_length_distribution(raw, cleaned, out_dir / "text_length_comparison.png")

#Plot 4: Similarity between raw text and cleaned text
def plot_similarity(merged_df, out_p, threshold=0.80):
    if merged_df is None:
        print("[WARN] merged dataframe not available for similarity analysis.")
        return

    raw_col = 'headline' if 'headline' in merged_df.columns else None
    cleaned_col = 'news_text' if 'news_text' in merged_df.columns else None
    if raw_col is None or cleaned_col is None:
        print("[WARN] Required columns for similarity not found. Needed 'headline' and 'news_text'.")
        return
    # compute similarity
    merged_df = merged_df.copy()
    merged_df['_sim'] = merged_df.apply(lambda r: similarity(r[raw_col], r[cleaned_col]), axis=1)
    merged_df['_equal'] = merged_df[raw_col].fillna("").astype(str) == merged_df[cleaned_col].fillna("").astype(str)

    total = len(merged_df)
    exact_equal = int(merged_df['_equal'].sum())
    similar_or_equal = int((merged_df['_sim'] >= threshold).sum())
    similar_but_not_equal = int(((merged_df['_sim'] >= threshold) & (~merged_df['_equal'])).sum())

    print(f"[INFO] similarity totals -> total={total}, exact_equal={exact_equal}, similar_or_equal(>={threshold})={similar_or_equal}, similar_but_not_equal={similar_but_not_equal}")

    sample = merged_df[(merged_df['_sim'] >= threshold) & (~merged_df['_equal'])]
    if not sample.empty:
        sample_path = out_dir / "similar_but_not_equal_sample.csv"
        sample.head(200).to_csv(sample_path, index=False)
        print(f"[SAVED] sample CSV: {sample_path}")

    # plot summary
    labels = ['exact_equal', 'similar_but_not_equal', 'not_similar']
    vals = [
        exact_equal,
        similar_but_not_equal,
        total - exact_equal - similar_but_not_equal
    ]
    plt.figure(figsize=(7,5))
    plt.bar(labels, vals)
    for i,v in enumerate(vals):
        plt.text(i, v + max(vals)*0.01, str(v), ha='center')
    plt.title(f"Similarity between '{raw_col}' and '{cleaned_col}' (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()
    print(f"[SAVED] {out_p}")

plot_similarity(merged, out_dir / "text_similarity.png")

print("\nAll done. Check the folder:", out_dir)


