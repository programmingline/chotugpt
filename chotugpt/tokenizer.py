"""ChotuGPT tokenizer.

Lecture 2: a character-level tokenizer — every unique character gets a number.
Lecture 15: a small byte-pair encoding (BPE) tokenizer, the same idea GPT-2 uses.
Lecture 22: save / load the trained tokenizer as JSON, and make training and encoding fast
            enough for a real corpus (count each distinct word once, remember encoded words).
"""
import json
import re


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))                            # every unique character, sorted
        self.stoi = {c: i for i, c in enumerate(self.chars)}      # character -> number
        self.itos = {i: c for c, i in self.stoi.items()}          # number -> character

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# split text into word-like chunks (" chai", ",", " feel") so merges never cross a word boundary
SPLIT = re.compile(r" ?[^\s,.!?]+| ?[,.!?]+|\s+")


def count_pairs(chunks, freqs=None):
    """How often each neighbouring pair appears. freqs[i] = how many times chunk i occurs (default 1)."""
    counts = {}
    for n, ids in enumerate(chunks):
        f = 1 if freqs is None else freqs[n]
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + f
    return counts


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)                                    # the pair becomes one new token
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE: start from the 256 possible bytes, then repeatedly merge the most frequent pair."""

    def __init__(self):
        self.merges = {}                                          # (id, id) -> new id, in the order learned
        self.vocab = {i: bytes([i]) for i in range(256)}          # id -> the bytes it stands for
        self.cache = {}                                           # word -> its token ids, filled by encode

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, num_merges, verbose=False):
        self.cache = {}                                           # new merges make old encodings stale
        # a corpus repeats the same words again and again: keep each distinct chunk once, with its count
        word_counts = {}
        for c in SPLIT.findall(text):
            word_counts[c] = word_counts.get(c, 0) + 1
        chunks = [list(c.encode("utf-8")) for c in word_counts]
        freqs = list(word_counts.values())
        for k in range(num_merges):
            counts = count_pairs(chunks, freqs)
            if not counts:
                break
            pair = max(counts, key=counts.get)                    # the most frequent pair
            new_id = 256 + k
            chunks = [merge(ids, pair, new_id) for ids in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {k + 1}: {self.vocab[new_id].decode('utf-8', 'replace')!r} ({counts[pair]} times)")

    def encode_chunk(self, c):
        if c not in self.cache:                                   # each distinct word is merged only once
            ids = list(c.encode("utf-8"))
            for pair, new_id in self.merges.items():              # replay the merges in the order learned
                ids = merge(ids, pair, new_id)
            self.cache[c] = ids
        return self.cache[c]

    def encode(self, text):
        out = []
        for c in SPLIT.findall(text):
            out.extend(self.encode_chunk(c))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text):
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]

    def save(self, path):
        """Only the merges need saving: the vocabulary can be rebuilt from them."""
        merges = [[a, b, new_id] for (a, b), new_id in self.merges.items()]
        with open(path, "w") as f:
            json.dump({"type": "byte-level BPE", "merges": merges}, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path) as f:
            for a, b, new_id in json.load(f)["merges"]:           # replay the merges in the saved order
                tok.merges[(a, b)] = new_id
                tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok


if __name__ == "__main__":
    text = "After my morning chai, I feel"
    tok = CharTokenizer(text)
    print(len(text), tok.vocab_size)        # 29 18
    print(tok.encode("chai"))               # [5, 9, 4, 10]
    print(tok.decode(tok.encode("chai")))   # chai

    corpus = "After my morning chai, I feel fresh. After lunch, I feel sleepy."
    bpe = BPETokenizer()
    bpe.train(corpus, 10)
    print(len(bpe.encode(text)))            # 18 tokens: only 10 merges, so many small pieces remain
    print(bpe.decode(bpe.encode(text)))     # After my morning chai, I feel
