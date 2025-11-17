# src/evaluate_merged.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PRED_CSV = os.path.join(BASE_DIR, "../data/processed/News_Category_Dataset_v3_predicted_merged.csv")
METRICS_DIR = os.path.join(BASE_DIR, "../results/metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "../results/figures")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------
# Category ↔ Label Mapping (merged categories)
# ----------------------------
category_to_label = {
    "BUSINESS": 0,
    "CRIME": 1,
    "EDUCATION": 2,
    "ENTERTAINMENT": 3,
    "ENVIRONMENT": 4,
    "FOOD & DRINK": 5,
    "IDENTITY/VOICES": 6,
    "LIFESTYLE": 7,
    "PARENTING": 8,
    "POLITICS": 9,
    "POSITIVE/HEALTH": 10,
    "SPORTS": 11,
    "STYLE/BEAUTY": 12,
    "TECH": 13,
    "TRAVEL": 14,
    "WORLD NEWS/GLOBAL": 15
}
label_to_category = {v: k for k, v in category_to_label.items()}

# ----------------------------
# Load predictions
# ----------------------------
df = pd.read_csv(PRED_CSV)

# Map true labels
df["category_label_id"] = df["category_merged"].map(category_to_label)

if df["category_label_id"].isna().sum() > 0:
    print("❗ Warning: Some merged categories were not mapped:")
    print(df[df["category_label_id"].isna()]["category_merged"].unique())

y_true = df["category_label_id"].astype(int).values
y_pred = df["pred_label_id"].astype(int).values

# ----------------------------
# Compute metrics
# ----------------------------
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "macro_f1": f1_score(y_true, y_pred, average="macro"),
    "micro_f1": f1_score(y_true, y_pred, average="micro"),
    "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    "macro_precision": precision_score(y_true, y_pred, average="macro"),
    "macro_recall": recall_score(y_true, y_pred, average="macro"),
    "weighted_precision": precision_score(y_true, y_pred, average="weighted"),
    "weighted_recall": recall_score(y_true, y_pred, average="weighted")
}

# ----------------------------
# Save main metrics
# ----------------------------
report = classification_report(y_true, y_pred, digits=4)
metrics_file = os.path.join(METRICS_DIR, "metrics_merged.txt")
with open(metrics_file, "w") as f:
    f.write("==== MODEL PERFORMANCE (MERGED CATEGORIES) ====\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("\n==== CLASSIFICATION REPORT ====\n")
    f.write(report)

print(f"✅ Metrics saved to {metrics_file}")

# ----------------------------
# Confusion matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", annot=True, fmt="d")
plt.title("Confusion Matrix (Merged Categories)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_merged.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, cmap="Blues", annot=True, fmt=".2f")
plt.title("Normalized Confusion Matrix (Merged Categories)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_normalized_merged.png"), dpi=300)
plt.close()

# ----------------------------
# Per-class metrics
# ----------------------------
report_dict = classification_report(y_true, y_pred, output_dict=True)
class_results = []

for label_id, metrics in report_dict.items():
    if label_id not in ["accuracy", "macro avg", "weighted avg", "micro avg"]:
        class_results.append([
            int(label_id),
            label_to_category[int(label_id)],
            metrics["precision"],
            metrics["recall"],
            metrics["f1-score"],
            metrics["support"]
        ])

class_df = pd.DataFrame(class_results, columns=["label_id", "category", "precision", "recall", "f1", "support"])
class_df.to_csv(os.path.join(METRICS_DIR, "per_class_metrics_merged.csv"), index=False)

# ----------------------------
# Per-class F1 bar chart
# ----------------------------
plt.figure(figsize=(14, 10))
class_df.sort_values("f1").plot(x="category", y="f1", kind="barh", legend=False)
plt.xlabel("F1 Score")
plt.title("F1 Score by Category (Merged)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "f1_score_per_class_merged.png"), dpi=300)
plt.close()

print(f"✅ Plots saved in {FIGURES_DIR}")
print("✅ Evaluation (merged categories) completed successfully!")
