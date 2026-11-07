"""ChotuGPT tokenizer.

Lecture 2: a character-level tokenizer — every unique character gets a number.
Lecture 15: a small byte-pair encoding (BPE) tokenizer, the same idea GPT-2 uses.
"""
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


def count_pairs(chunks):
    counts = {}
    for ids in chunks:
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
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

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, num_merges, verbose=False):
        chunks = [list(c.encode("utf-8")) for c in SPLIT.findall(text)]
        for k in range(num_merges):
            counts = count_pairs(chunks)
            if not counts:
                break
            pair = max(counts, key=counts.get)                    # the most frequent pair
            new_id = 256 + k
            chunks = [merge(ids, pair, new_id) for ids in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {k + 1}: {self.vocab[new_id].decode('utf-8', 'replace')!r} ({counts[pair]} times)")

    def encode(self, text):
        out = []
        for c in SPLIT.findall(text):
            ids = list(c.encode("utf-8"))
            for pair, new_id in self.merges.items():              # replay the merges in the order learned
                ids = merge(ids, pair, new_id)
            out.extend(ids)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text):
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]


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
