"""ChotuGPT pretraining data.

Lecture 23: clean the raw text, split it into train / validation, train the BPE tokenizer
and turn the text into token IDs saved as small binary files.

    raw text -> strip tags -> drop short lines -> drop noise -> remove duplicates
             -> train/val split (90/10) -> BPE tokenizer (512 tokens) -> train.bin / val.bin
"""
import os
import random
import re

import numpy as np

from chotugpt.tokenizer import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
VOCAB_SIZE = 512                       # 256 bytes + 256 merges
MIN_CHARS = 20                         # lines shorter than this are scraps, not sentences
TAG = re.compile(r"<[^>]+>")           # anything that looks like <p>, </div>, <br> ...


def path(name):
    return os.path.normpath(os.path.join(DATA_DIR, name))


# ---- cleaning: each step is one small function -------------------------------------------

def strip_tags(lines):
    """Remove HTML-like tags but keep the text inside them."""
    return [" ".join(TAG.sub(" ", line).split()) for line in lines]   # also tidies the spaces


def drop_short(lines, min_chars=MIN_CHARS):
    return [line for line in lines if len(line) >= min_chars]


def is_noise(line):
    """Mostly symbols and digits instead of letters? Then it is not text."""
    letters = sum(ch.isalpha() or ch == " " for ch in line)
    return letters / len(line) < 0.7


def drop_noise(lines):
    return [line for line in lines if not is_noise(line)]


def deduplicate(lines):
    """Exact-line deduplication: keep the first copy of every line, in order."""
    seen, out = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def train_val_split(lines, val_frac=0.1, seed=23):
    """Shuffle the lines (with a fixed seed), keep 90% for training and 10% to check on."""
    lines = list(lines)
    random.Random(seed).shuffle(lines)
    n_val = int(len(lines) * val_frac)
    return lines[n_val:], lines[:n_val]


# ---- tokenizer and binary files -----------------------------------------------------------

def train_tokenizer(text, vocab_size=VOCAB_SIZE):
    tok = BPETokenizer()
    tok.train(text, num_merges=vocab_size - 256)
    return tok


def load_tokenizer():
    return BPETokenizer.load(path("tokenizer.json"))


def save_ids(ids, filename):
    """Token IDs < 65536 fit in 2 bytes each (uint16): a compact file the trainer can read fast."""
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path(filename))
    return arr


def load_ids(split):
    """split = "train" or "val" -> a 1-D array of token IDs."""
    return np.fromfile(path(f"{split}.bin"), dtype=np.uint16)


def prepare(corpus="chai_corpus.txt"):
    """The whole pipeline. Returns a dict of what happened at every step (lecture 23 prints it)."""
    with open(path(corpus), encoding="utf-8") as f:
        raw = f.read().splitlines()               # one diary entry (or one piece of junk) per line
    steps = [("raw", raw)]
    lines = strip_tags(raw);    steps.append(("strip tags", lines))
    lines = drop_short(lines);  steps.append(("drop short lines", lines))
    lines = drop_noise(lines);  steps.append(("drop symbol noise", lines))
    lines = deduplicate(lines); steps.append(("remove duplicates", lines))

    train_lines, val_lines = train_val_split(lines)
    train_text, val_text = "\n".join(train_lines) + "\n", "\n".join(val_lines) + "\n"

    tok = train_tokenizer(train_text)               # learn merges from the training text only
    tok.save(path("tokenizer.json"))
    train_ids = save_ids(tok.encode(train_text), "train.bin")
    val_ids = save_ids(tok.encode(val_text), "val.bin")
    return {"steps": steps, "train_lines": train_lines, "val_lines": val_lines,
            "train_chars": len(train_text), "val_chars": len(val_text),
            "tokenizer": tok, "train_ids": train_ids, "val_ids": val_ids}
