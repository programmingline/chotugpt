"""ChotuGPT tokenizer.

Lecture 2: a character-level tokenizer — every unique character gets a number.
(Lecture 15 replaces this with byte-pair encoding.)
"""


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


if __name__ == "__main__":
    text = "After my morning chai, I feel"
    tok = CharTokenizer(text)
    print(len(text), tok.vocab_size)        # 29 18
    print(tok.encode("chai"))               # [5, 9, 4, 10]
    print(tok.decode(tok.encode("chai")))   # chai
