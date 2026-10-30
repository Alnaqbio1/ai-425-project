import re
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HF_BASE = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text"
SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMDB_DIR = DATA_DIR / "imdb"

RANDOM_SEED = 42
N_PER_CLASS = 2000
VAL_FRACTION = 0.20


def download_imdb(imdb_dir=IMDB_DIR):
    """Download the official train and test splits if they are not already there."""
    imdb_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in SPLIT_FILES.items():
        target = imdb_dir / f"{split}.parquet"
        if not target.exists():
            print("Downloading IMDb {} split ...".format(split))
            urllib.request.urlretrieve(f"{HF_BASE}/{filename}", target)
    return imdb_dir


def load_split(split="train"):
    df = pd.read_parquet(IMDB_DIR / f"{split}.parquet")[["text", "label"]]
    df.insert(0, "review_id", range(len(df)))
    return df


def balanced_subset(df, n_per_class=N_PER_CLASS, seed=RANDOM_SEED):
    parts = [
        df[df["label"] == label].sample(n=n_per_class, random_state=seed)
        for label in (0, 1)
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def prepare_data(n_per_class=N_PER_CLASS, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    download_imdb()
    full_train = load_split("train")
    subset = balanced_subset(full_train, n_per_class, seed)
    subset["clean_text"] = subset["text"].map(clean_text)
    train_df, val_df = train_test_split(
        subset, test_size=val_fraction, stratify=subset["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def training_pool(val_df, n_total=None, seed=RANDOM_SEED):
    full = load_split("train")
    pool = full[~full["review_id"].isin(val_df["review_id"])]
    if n_total is not None:
        pool = balanced_subset(pool, n_total // 2, seed)
    pool = pool.copy()
    pool["clean_text"] = pool["text"].map(clean_text)
    return pool.reset_index(drop=True)


def prepare_test():
    download_imdb()
    test = load_split("test")
    test["clean_text"] = test["text"].map(clean_text)
    return test


EXTRA_DIR = DATA_DIR / "extra"
ROTTEN_URL = "https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes/resolve/main/test.parquet"
AMAZON_URL = "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/main/en/test.jsonl"


def _fetch(url, target):
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print("Downloading {} ...".format(target.name))
        urllib.request.urlretrieve(url, target)
    return target


def load_rotten_tomatoes():
    df = pd.read_parquet(_fetch(ROTTEN_URL, EXTRA_DIR / "rotten_tomatoes_test.parquet"))
    df["clean_text"] = df["text"].map(clean_text)
    return df[["text", "clean_text", "label"]]


def load_amazon_reviews():
    df = pd.read_json(_fetch(AMAZON_URL, EXTRA_DIR / "amazon_reviews_en_test.jsonl"), lines=True)
    stars = df["label"] + 1
    df = df[stars != 3].copy()
    df["label"] = (stars[stars != 3] >= 4).astype(int)
    df["clean_text"] = df["text"].map(clean_text)
    return df[["text", "clean_text", "label"]].reset_index(drop=True)


def load_challenge_set(path=DATA_DIR / "challenge_negation.csv"):
    df = pd.read_csv(path)
    df["label_id"] = (df["label"] == "positive").astype(int)
    df["clean_text"] = df["text"].map(clean_text)
    return df


if __name__ == "__main__":
    train_df, val_df = prepare_data()
    print("train: {} reviews, {:.0%} positive".format(len(train_df), train_df['label'].mean()))
    print("val:   {} reviews, {:.0%} positive".format(len(val_df), val_df['label'].mean()))
