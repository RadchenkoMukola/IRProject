# predict.py  (local-files-only, simple config at top)
import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
import torch.nn.functional as F
from pathlib import Path
# ----------------------------
# Project root (as you had)
# ----------------------------
os.chdir("..")

hf_logging.set_verbosity_error()  # quiet HF warnings

# === CONFIGURATION (edit these) ===
MODEL_DIR = "./results/models/best_BERT_model"   # <--- your local model folder
INPUT_CSV = "./Data/processed/News_Category_Dataset_v3_cleaned.csv"
OUTPUT_CSV = "./Data/processed/News_Category_Dataset_v3_predicted.csv"
TEXT_COL = "news_text"
BATCH_SIZE = 64
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FALLBACK_TOKENIZER = "bert-base-uncased"  # used if tokenizer files missing in MODEL_DIR
# ===================================


def load_model_and_tokenizer_local(model_dir: str, fallback_tokenizer: str = None):
    model_dir = str(Path(model_dir).resolve())
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    print(f"Loading model from (local only): {model_dir}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_dir} locally: {e}")

    # load tokenizer from saved folder (local only) or fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        print("Tokenizer loaded from model folder.")
    except Exception as e:
        if fallback_tokenizer:
            print(f"[warning] tokenizer not found in {model_dir} (or failed to load). Loading fallback tokenizer '{fallback_tokenizer}' from hub/local cache.")
            tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer, use_fast=True, local_files_only=False)
        else:
            raise RuntimeError(f"Failed to load tokenizer from {model_dir} and no fallback provided: {e}")
    return model, tokenizer


def batch_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(it))
        except StopIteration:
            if batch:
                yield batch
            break
        yield batch


def main():
    print(f"Using device: {DEVICE}")

    model, tokenizer = load_model_and_tokenizer_local(MODEL_DIR, FALLBACK_TOKENIZER)
    model.to(DEVICE)
    model.eval()

    # try to obtain id2label from config
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        try:
            id2label = {int(k): v for k, v in id2label.items()}
            print(f"Loaded id2label mapping with {len(id2label)} labels.")
        except Exception:
            # leave as-is if can't convert
            print("Model config contains id2label but keys couldn't be converted to int; using raw mapping.")
    else:
        print("No id2label mapping found in model config; predictions will be numeric ids.")

    # load CSV
    df = pd.read_csv(INPUT_CSV)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in {INPUT_CSV}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    non_empty = [(i, txt) for i, txt in enumerate(df[TEXT_COL]) if txt.strip()]
    print(f"Found {len(non_empty)} non-empty texts out of {len(df)} rows.")

    pred_ids = np.full(len(df), -1, dtype=int)
    pred_scores = np.full(len(df), np.nan)
    pred_labels = np.full(len(df), "", dtype=object)

    start = time.time()
    for batch in tqdm(batch_iterable(non_empty, BATCH_SIZE),
                      total=(len(non_empty) + BATCH_SIZE - 1) // BATCH_SIZE):
        idxs = [b[0] for b in batch]
        texts = [b[1] for b in batch]

        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = probs.max(dim=-1)

        top_probs = top_probs.cpu().numpy()
        top_ids = top_ids.cpu().numpy()

        for i, idx in enumerate(idxs):
            pred_ids[idx] = int(top_ids[i])
            pred_scores[idx] = float(top_probs[i])
            if id2label:
                pred_labels[idx] = id2label.get(pred_ids[idx], str(pred_ids[idx]))
            else:
                pred_labels[idx] = str(pred_ids[idx])

    elapsed = time.time() - start
    print(f"Inference finished in {elapsed:.1f}s — {len(non_empty) / elapsed:.1f} samples/s")

    df["pred_label_id"] = pred_ids
    df["pred_label"] = pred_labels
    df["pred_score"] = pred_scores

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Predictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
