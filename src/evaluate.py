# eval.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
os.chdir("..")


# ----------------------------
# File paths
# ----------------------------
PRED_CSV = "./Data/processed/News_Category_Dataset_v3_predicted.csv"

# ----------------------------
# Training label mapping (category -> label_id)
# ----------------------------
category_to_label = {
    'ARTS': 0, 'ARTS & CULTURE': 1, 'BLACK VOICES': 2, 'BUSINESS': 3, 'COLLEGE': 4,
    'COMEDY': 5, 'CRIME': 6, 'CULTURE & ARTS': 7, 'DIVORCE': 8, 'EDUCATION': 9,
    'ENTERTAINMENT': 10, 'ENVIRONMENT': 11, 'FIFTY': 12, 'FOOD & DRINK': 13, 'GOOD NEWS': 14,
    'GREEN': 15, 'HEALTHY LIVING': 16, 'HOME & LIVING': 17, 'IMPACT': 18, 'LATINO VOICES': 19,
    'MEDIA': 20, 'MONEY': 21, 'PARENTING': 22, 'PARENTS': 23, 'POLITICS': 24,
    'QUEER VOICES': 25, 'RELIGION': 26, 'SCIENCE': 27, 'SPORTS': 28, 'STYLE': 29,
    'STYLE & BEAUTY': 30, 'TASTE': 31, 'TECH': 32, 'THE WORLDPOST': 33, 'TRAVEL': 34,
    'U.S. NEWS': 35, 'WEDDINGS': 36, 'WEIRD NEWS': 37, 'WELLNESS': 38, 'WOMEN': 39,
    'WORLD NEWS': 40, 'WORLDPOST': 41
}

# ----------------------------
# Load predicted CSV
# ----------------------------
df = pd.read_csv(PRED_CSV)

# Map categories to numeric label IDs
df['category_label_id'] = df['category'].map(category_to_label)

# Check for unmapped categories
missing = df[df['category_label_id'].isna()]
if not missing.empty:
    print("Warning! Some categories are not in training mapping:")
    print(missing['category'].unique())

# ----------------------------
# Evaluation
# ----------------------------
y_true = df['category_label_id'].astype(int).values
y_pred = df['pred_label_id'].astype(int).values

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# ----------------------------
# Confusion matrix plot
# ----------------------------
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# Save mapped CSV for reference
# ----------------------------
df.to_csv(PRED_CSV.replace(".csv", "_mapped.csv"), index=False)
print("Mapped CSV saved.")
