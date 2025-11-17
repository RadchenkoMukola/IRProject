import pandas as pd
import os

os.chdir("..")  # move to project root
# Paths
CLEANED_CSV = "Data/processed/News_Category_Dataset_v3_cleaned.csv"
MERGED_CSV = "Data/processed/News_Category_Dataset_v3_merged.csv"
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
print("\nNaN counts per column (cleaned):")
print(df_cleaned.isna().sum())

print("\nNaN counts per column (original):")
print(df_original.isna().sum())

# -------------------
# Remove rows with NaN in 'news_text' or 'headline'
# -------------------
before_removal = len(df_cleaned)
df_cleaned = df_cleaned.dropna(subset=['news_text', 'headline'])
after_removal = len(df_cleaned)

print(f"\nRows removed due to NaN in 'news_text' or 'headline': {before_removal - after_removal}")
print(f"Remaining rows in cleaned dataset: {after_removal}")

# -------------------
# Merge categories into 15 classes
# -------------------
merge_map = {
    # POLITICS
    'POLITICS': 'POLITICS',

    # Positive / Health / Misc
    'POSITIVE/HEALTH': 'POSITIVE/HEALTH',
    'GOOD NEWS': 'POSITIVE/HEALTH',
    'IMPACT': 'POSITIVE/HEALTH',
    'HEALTHY LIVING': 'POSITIVE/HEALTH',
    'WELLNESS': 'POSITIVE/HEALTH',
    'WEIRD NEWS': 'POSITIVE/HEALTH',

    # Entertainment & Arts
    'ENTERTAINMENT': 'ENTERTAINMENT',
    'COMEDY': 'ENTERTAINMENT',
    'ARTS': 'ENTERTAINMENT',
    'ARTS & CULTURE': 'ENTERTAINMENT',
    'CULTURE & ARTS': 'ENTERTAINMENT',

    # Identity / Voices
    'BLACK VOICES': 'IDENTITY/VOICES',
    'LATINO VOICES': 'IDENTITY/VOICES',
    'QUEER VOICES': 'IDENTITY/VOICES',
    'WOMEN': 'IDENTITY/VOICES',
    'RELIGION': 'IDENTITY/VOICES',

    # Parenting
    'PARENTING': 'PARENTING',
    'PARENTS': 'PARENTING',

    # Style / Beauty
    'STYLE': 'STYLE/BEAUTY',
    'STYLE & BEAUTY': 'STYLE/BEAUTY',

    # Travel
    'TRAVEL': 'TRAVEL',

    # Food / Drink
    'FOOD & DRINK': 'FOOD & DRINK',
    'TASTE': 'FOOD & DRINK',

    # Business
    'BUSINESS': 'BUSINESS',
    'MONEY': 'BUSINESS',

    # World / Global news
    'WORLD NEWS': 'WORLD NEWS/GLOBAL',
    'WORLDPOST': 'WORLD NEWS/GLOBAL',
    'U.S. NEWS': 'WORLD NEWS/GLOBAL',
    'THE WORLDPOST': 'WORLD NEWS/GLOBAL',

    # Sports
    'SPORTS': 'SPORTS',


    # Lifestyle
    'HOME & LIVING': 'LIFESTYLE',
    'WEDDINGS': 'LIFESTYLE',
    'DIVORCE': 'LIFESTYLE',
    'FIFTY': 'LIFESTYLE',

    # Education
    'EDUCATION': 'EDUCATION',
    'COLLEGE': 'EDUCATION',

    # Environment
    'ENVIRONMENT': 'ENVIRONMENT',
    'GREEN': 'ENVIRONMENT',

    # Tech & Media
    'TECH': 'TECH',
    'SCIENCE': 'TECH',
    'MEDIA': 'TECH'

}

# Apply mapping
df_cleaned['category_merged'] = df_cleaned['category'].map(lambda x: merge_map.get(x, x))

# -------------------
# Report merged category counts
# -------------------
print("\nMerged category counts:")
print(df_cleaned['category_merged'].value_counts().sort_values(ascending=False))

# -------------------
# Save merged dataset
# -------------------
df_cleaned.to_csv(MERGED_CSV, index=False)
print(f"\nMerged dataset saved to: {MERGED_CSV}")
