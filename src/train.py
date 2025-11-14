# bert_news_pipeline_stable_tokenizer_fixed_with_resume.py
# Stable tokenization: use fast tokenizer, smaller chunks, robust fallbacks & logging
# + automatic checkpoint detection, model load from checkpoint, and resume training

import os
import gc
import pickle
import logging
from typing import List, Dict
import pandas as pd
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

import threading
import torch.nn as nn

_old_train = nn.Module.train
_reentrant = threading.local()

def _safe_train(self, mode: bool = True):
    # re-entrancy guard: if we are already inside safe_train for this thread,
    # call the original implementation directly to avoid recursion.
    if getattr(_reentrant, "in_safe_train", False):
        return _old_train(self, mode)

    _reentrant.in_safe_train = True
    try:
        try:
            # first attempt: normal call
            return _old_train(self, mode)
        except NameError as ne:
            # original closure bug: missing __class__ in closure
            if "__class__" in str(ne):
                try:
                    # ensure the class binding exists
                    self.__class__ = type(self)
                except Exception:
                    # if we can't set it, still try calling old_train again
                    pass
                return _old_train(self, mode)
            raise
        except SystemError:
            # C-level error path (some PyTorch internals can raise SystemError).
            # Try to rebind class and retry once.
            try:
                self.__class__ = type(self)
            except Exception:
                pass
            return _old_train(self, mode)
    finally:
        _reentrant.in_safe_train = False

# Apply the monkey-patch
nn.Module.train = _safe_train

from transformers.trainer_utils import get_last_checkpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Project root (as you had)
# ----------------------------
os.chdir("..")

# ----------------------------
# Clear memory
# ----------------------------
torch.cuda.empty_cache()
gc.collect()

# ----------------------------
# Load dataset
# ----------------------------
DATA_PATH = "Data/processed/News_Category_Dataset_v3_cleaned.csv"
df = pd.read_csv(DATA_PATH, dtype={"news_text": str, "category": str}, low_memory=False)
df = df.dropna(subset=["news_text", "category"])

le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])
texts = df["news_text"].astype(str).tolist()
labels = df["label"].tolist()
num_labels = df["label"].nunique()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# ----------------------------
# Tokenizer: USE THE FAST TOKENIZER
# ----------------------------
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
MAX_LENGTH = 256  # shorter length for speed/stability


# ----------------------------
# Robust chunked tokenization with fallbacks (unchanged)
# ----------------------------
def safe_chunked_tokenize(
        texts: List[str],
        tokenizer,
        max_length: int = 256,
        cache_path: str = "cache.pkl",
        chunk_size: int = 1000,
        max_text_chars: int = 50_000
) -> Dict[str, List]:
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            logger.info(f"Loaded cached encodings from {cache_path}")
            return pickle.load(f)

    total = len(texts)
    all_input_ids = []
    all_attention_masks = []
    bad_indices = []

    logger.info(f"Tokenizing {total:,} samples in chunks of {chunk_size}")

    def _truncate_texts(batch_texts):
        truncated = []
        for t in batch_texts:
            if t is None:
                truncated.append("")
            elif len(t) > max_text_chars:
                truncated.append(t[:max_text_chars])
            else:
                truncated.append(t)
        return truncated

    i = 0
    while i < total:
        j = min(i + chunk_size, total)
        batch_texts = texts[i:j]
        batch_texts = _truncate_texts(batch_texts)

        try:
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True
            )
            all_input_ids.extend(enc["input_ids"])
            all_attention_masks.extend(enc["attention_mask"])

            # Save progress
            with open(cache_path, "wb") as f:
                pickle.dump({"input_ids": all_input_ids, "attention_mask": all_attention_masks}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"[{j}/{total}] Tokenized chunk (size={j - i})")
            del enc
            gc.collect()
            i = j

        except Exception as e:
            logger.warning(
                f"Batch tokenization failed for indices [{i}:{j}] with exception: {e}. Trying smaller batches.")
            if j - i == 1:
                logger.error(f"Single item tokenization failed at index {i}. Logging and skipping.")
                bad_indices.append(i)
                input_len = max_length
                all_input_ids.append([tokenizer.pad_token_id or 0] * input_len)
                all_attention_masks.append([0] * input_len)
                i += 1
                with open(cache_path, "wb") as f:
                    pickle.dump({"input_ids": all_input_ids, "attention_mask": all_attention_masks}, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                continue

            sub_batch_size = max(1, (j - i) // 4)
            k = i
            while k < j:
                l = min(k + sub_batch_size, j)
                sub_texts = texts[k:l]
                sub_texts = _truncate_texts(sub_texts)
                try:
                    sub_enc = tokenizer(
                        sub_texts,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_attention_mask=True
                    )
                    all_input_ids.extend(sub_enc["input_ids"])
                    all_attention_masks.extend(sub_enc["attention_mask"])
                    logger.info(f"[{l}/{total}] Tokenized sub-chunk (size={l - k})")
                    del sub_enc
                    gc.collect()
                    k = l
                    with open(cache_path, "wb") as f:
                        pickle.dump({"input_ids": all_input_ids, "attention_mask": all_attention_masks}, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e2:
                    logger.warning(f"Sub-batch [{k}:{l}] failed ({e2}). Falling back to item-by-item.")
                    for idx in range(k, l):
                        try:
                            txt = texts[idx]
                            if txt is None:
                                txt = ""
                            if len(txt) > max_text_chars:
                                txt = txt[:max_text_chars]
                            single_enc = tokenizer(
                                txt,
                                truncation=True,
                                padding="max_length",
                                max_length=max_length,
                                return_attention_mask=True
                            )
                            all_input_ids.append(single_enc["input_ids"])
                            all_attention_masks.append(single_enc["attention_mask"])
                            logger.info(f"[{idx + 1}/{total}] Tokenized single item fallback.")
                            del single_enc
                            gc.collect()
                        except Exception as e3:
                            logger.error(f"Single item tokenization failed at index {idx}: {e3}. Adding placeholder.")
                            bad_indices.append(idx)
                            all_input_ids.append([tokenizer.pad_token_id or 0] * max_length)
                            all_attention_masks.append([0] * max_length)
                        with open(cache_path, "wb") as f:
                            pickle.dump({"input_ids": all_input_ids, "attention_mask": all_attention_masks}, f,
                                        protocol=pickle.HIGHEST_PROTOCOL)
                    k = l
            i = j

    if bad_indices:
        bad_log_path = cache_path.replace(".pkl", "_bad_indices.txt")
        with open(bad_log_path, "w", encoding="utf-8") as f:
            for idx in bad_indices:
                f.write(f"{idx}\n")
        logger.warning(
            f"Some inputs failed tokenization and were replaced with placeholders. Indices saved to {bad_log_path}")

    logger.info(f"Finished tokenization; final size = {len(all_input_ids)}")
    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks}


# Tokenize train & val (safe)
train_encodings = safe_chunked_tokenize(train_texts, tokenizer, MAX_LENGTH, "bert_train_cache_fast.pkl",
                                        chunk_size=1000)
val_encodings = safe_chunked_tokenize(val_texts, tokenizer, MAX_LENGTH, "bert_val_cache_fast.pkl", chunk_size=1000)


# ----------------------------
# Dataset & rest
# ----------------------------
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=np.array(train_labels))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ----------------------------
# Check for checkpoint and load model weights if present
# ----------------------------
OUTPUT_DIR = "./results_bert_weighted"
last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
if last_checkpoint is not None:
    logger.info(f"Found checkpoint: {last_checkpoint}. Loading model weights from checkpoint.")
    model = BertForSequenceClassification.from_pretrained(last_checkpoint, num_labels=num_labels)
else:
    logger.info("No checkpoint found. Loading model from pretrained model name.")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


# ----------------------------
# Weighted Trainer (only override compute_loss)
# ----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weight = self.class_weights.to(logits.dtype).to(model.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ----------------------------
# Training arguments (improve checkpoint behavior)
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs_bert_weighted",
    logging_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=1e-5,
    fp16=True,
    max_grad_norm=5.0,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor / class_weights_tensor.mean(),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("Device name: None")

torch.cuda.empty_cache()
gc.collect()

# ----------------------------
# Start training: attempt resume if checkpoint exists
# ----------------------------
if last_checkpoint is None:
    logger.info("Starting training from scratch.")
    trainer.train()
else:
    logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    # This will restore optimizer/scheduler state and resume steps/epochs
    trainer.train(resume_from_checkpoint=last_checkpoint)
