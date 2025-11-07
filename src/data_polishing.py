import pandas as pd

# Paths
CLEANED_CSV = "Data/processed/News_Category_Dataset_v3_cleaned.csv"
ORIGINAL_CSV = "Data/raw/News_Category_Dataset_v3.csv"

# Load datasets
df_cleaned = pd.read_csv(CLEANED_CSV)
df_original = pd.read_csv(ORIGINAL_CSV)

# -------------------
# Report row counts
# -------------------
print(f"Cleaned dataset rows: {len(df_cleaned)}")
print(f"Original dataset rows: {len(df_original)}")

# -------------------
# Count NaN rows per column
# -------------------
cleaned_nan_counts = df_cleaned.isna().sum()
original_nan_counts = df_original.isna().sum()

print("\nNaN counts per column (cleaned):")
print(cleaned_nan_counts)

print("\nNaN counts per column (original):")
print(original_nan_counts)

# -------------------
# Compare specific columns
# -------------------
if 'news_text' in df_cleaned.columns:
    cleaned_empty_news = df_cleaned['news_text'].isna().sum()
    print(f"\nEmpty 'news_text' rows in cleaned: {cleaned_empty_news}")

if 'short_description' in df_original.columns:
    original_empty_short_desc = df_original['short_description'].isna().sum()
    print(f"Empty 'short_description' rows in original: {original_empty_short_desc}")

if 'headline' in df_cleaned.columns:
    cleaned_empty_headline = df_cleaned['headline'].isna().sum()
    print(f"Empty 'headline' rows in cleaned: {cleaned_empty_headline}")

# -------------------
# Remove rows with NaN in 'news_text' or 'headline'
# -------------------
before_removal = len(df_cleaned)
df_cleaned = df_cleaned.dropna(subset=['news_text', 'headline'])
after_removal = len(df_cleaned)

print(f"\nRows removed due to NaN in 'news_text' or 'headline': {before_removal - after_removal}")
print(f"Remaining rows in cleaned dataset: {after_removal}")

# -------------------
# Save updated cleaned dataset
# -------------------
df_cleaned.to_csv(CLEANED_CSV, index=False)
print(f"Updated cleaned dataset saved to: {CLEANED_CSV}")

