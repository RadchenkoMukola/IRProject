"""
clean_and_extract_newspaper.py

What it does:
- Loads dataset (JSON lines)
- Removes columns: short_description, authors, date (if present)
- Extracts full article text using newspaper3k (if a URL column exists)
- Saves cleaned CSV with a 'news_text' column (may be None if extraction fails)

Requirements:
pip install pandas newspaper3k tqdm
python -c "import nltk; nltk.download('punkt')"  # run once for newspaper3k
"""
from __future__ import annotations




# clean_and_extract_news.py
"""
Improved pipeline:
- Uses newspaper3k + trafilatura for robust extraction
- Preserves checkpoint discovery & reconciliation from original
- Cleans boilerplate (ads, LOADING ERROR)
- Logs failed URLs for retry
"""


import os
import asyncio
import async_timeout
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from newspaper import Article, Config
import trafilatura
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
import time
import glob
import sys

# -----------------------------
# Configuration
# -----------------------------
os.chdir("..")  # move to project root
ROOT = Path.cwd()
INPUT_PATH = ROOT / "Data/raw/News_Category_Dataset_v3.json"
OUTPUT_PATH = ROOT / "Data/processed/News_Category_Dataset_v3_cleaned.csv"
CHECKPOINT_DIR = ROOT / "Data/processed"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS_TO_REMOVE = ["short_description", "authors", "date"]

# Performance
BATCH_SIZE = 5000
MAX_CONNECTIONS = 200
MAX_CONCURRENT_FETCHES = 200
MAX_PARSER_THREADS = 16
REQUEST_TIMEOUT = 20
RETRIES = 2
SLEEP_BACKOFF = 1.0
USER_AGENT = "Mozilla/5.0 (compatible; NewsScraper/1.0)"

config = Config()
config.request_timeout = REQUEST_TIMEOUT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# Utilities
# -----------------------------
def find_latest_checkpoint(pattern_base: str) -> Path | None:
    pattern = str(CHECKPOINT_DIR / f"{pattern_base}.checkpoint.*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0])

def load_checkpoint_if_exists(base_stem: str) -> pd.DataFrame | None:
    cp = find_latest_checkpoint(base_stem)
    if cp is None:
        return None
    logging.info(f"Loading checkpoint: {cp}")
    try:
        return pd.read_csv(cp, dtype={'news_text': str}, low_memory=False)
    except Exception:
        logging.exception("Failed to load checkpoint file.")
        return None

def clean_text(text: str) -> str | None:
    if not text:
        return None
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(phrase in line.lower() for phrase in ["advertisement", "your support", "loading error"]):
            continue
        if len(line) < 30:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() or None

def extract_text(url: str, html: str | None = None) -> str | None:
    # Newspaper3k
    if html:
        try:
            art = Article(url, config=config)
            art.set_html(html)
            art.parse()
            text = art.text.strip()
            if text:
                return clean_text(text)
        except Exception:
            pass
    # Trafilatura fallback
    if html is None:
        downloaded = trafilatura.fetch_url(url)
    else:
        downloaded = html
    if downloaded:
        text = trafilatura.extract(downloaded)
        if text:
            return clean_text(text)
    return None

def is_valid_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    url = url.strip().lower()
    if not url.startswith("http"):
        return False
    if any(domain in url for domain in ["twitter.com", "youtube.com", "t.me"]):
        return False
    return True

# -----------------------------
# Async fetch + parse
# -----------------------------
async def fetch_html(session, url, timeout=REQUEST_TIMEOUT, retries=RETRIES):
    backoff = SLEEP_BACKOFF
    for attempt in range(1, retries + 1):
        try:
            async with async_timeout.timeout(timeout):
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == retries:
                return None
            await asyncio.sleep(backoff)
            backoff *= 2
    return None

async def process_batch(index_url_pairs, executor, session, sem):
    loop = asyncio.get_running_loop()
    results: dict = {}
    async def do_one(idx, url):
        async with sem:
            if not is_valid_url(url):
                results[idx] = None
                return
            html = await fetch_html(session, url)
        text = await loop.run_in_executor(executor, extract_text, url, html)
        results[idx] = text
    tasks = [asyncio.create_task(do_one(idx, url)) for idx, url in index_url_pairs]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="fetch+parse", leave=False):
        await f
    return results

async def run_extraction_for_indices(df, url_col, remaining_indices):
    connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
    timeout = aiohttp.ClientTimeout(total=None)
    headers = {"User-Agent": USER_AGENT}
    sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    executor = ThreadPoolExecutor(max_workers=MAX_PARSER_THREADS)
    collected = {}
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
        total = len(remaining_indices)
        batch_count = (total + BATCH_SIZE - 1) // BATCH_SIZE
        logging.info(f"{total} rows to process in {batch_count} batch(es)")
        for b in range(batch_count):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, total)
            batch_indices = remaining_indices[start:end]
            index_url_pairs = [(idx, df.at[idx, url_col]) for idx in batch_indices]
            logging.info(f"Batch {b+1}/{batch_count}: indices {batch_indices[0]}..{batch_indices[-1]}")
            batch_results = await process_batch(index_url_pairs, executor, session, sem)
            collected.update(batch_results)
            # checkpoint
            df.loc[pd.Index(collected.keys()), 'news_text'] = pd.Series(collected)
            ts = time.strftime("%Y%m%dT%H%M%S")
            checkpoint_file = CHECKPOINT_DIR / f"{OUTPUT_PATH.stem}.checkpoint.batch{b+1}.{ts}.csv"
            df.to_csv(checkpoint_file, index=False)
            logging.info(f"Checkpoint saved: {checkpoint_file}")
    executor.shutdown(wait=True)
    return collected

# -----------------------------
# Main
# -----------------------------
def main():
    logging.info("Loading dataset...")
    df = pd.read_json(INPUT_PATH, lines=True)
    df.drop(columns=[c for c in COLUMNS_TO_REMOVE if c in df.columns], inplace=True)
    logging.info(f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")
    url_col_candidates = [c for c in df.columns if "url" in c.lower() or "link" in c.lower()]
    if not url_col_candidates:
        logging.info("No URL column found. Saving cleaned CSV without news_text.")
        df.to_csv(OUTPUT_PATH, index=False)
        return
    url_col = url_col_candidates[0]
    logging.info(f"Using URL column: {url_col}")
    df[url_col] = df[url_col].fillna("").astype(str)
    if 'news_text' not in df.columns:
        df['news_text'] = None

    # Load checkpoint if exists
    checkpoint_df = load_checkpoint_if_exists(OUTPUT_PATH.stem)
    if checkpoint_df is not None:
        if len(checkpoint_df) == len(df):
            df['news_text'] = checkpoint_df.get('news_text')
            logging.info("Checkpoint aligns with dataset length. Adopting 'news_text'.")
        else:
            logging.info("Checkpoint length differs. Attempting URL-based reconciliation.")
            if url_col in checkpoint_df.columns:
                url_to_text = dict(zip(checkpoint_df[url_col].astype(str), checkpoint_df.get('news_text')))
                df['news_text'] = df[url_col].map(lambda u: url_to_text.get(str(u), None))
            else:
                logging.warning("Checkpoint has no URL column; cannot reconcile. Starting fresh.")

    # Determine remaining rows
    done_mask = df['news_text'].apply(lambda x: pd.notna(x) and str(x).strip() != "" and str(x).strip().lower() != "none")
    remaining_indices = df.index[~done_mask].tolist()
    logging.info(f"{len(df) - len(remaining_indices)} rows already done; {len(remaining_indices)} remaining")

    if not remaining_indices:
        logging.info("All rows completed. Saving final CSV.")
        df.to_csv(OUTPUT_PATH, index=False)
        return

    try:
        collected = asyncio.run(run_extraction_for_indices(df, url_col, remaining_indices))
        if collected:
            df.loc[pd.Index(collected.keys()), 'news_text'] = pd.Series(collected)
        df.to_csv(OUTPUT_PATH, index=False)
        logging.info(f"✅ Final cleaned dataset saved to {OUTPUT_PATH}")
    except KeyboardInterrupt:
        logging.warning("Interrupted. Saving partial progress.")
        df.to_csv(OUTPUT_PATH.with_suffix(".partial.csv"), index=False)
    except Exception:
        logging.exception("Unhandled exception. Saving partial progress.")
        df.to_csv(OUTPUT_PATH.with_suffix(".error.partial.csv"), index=False)

if __name__ == "__main__":
    main()
