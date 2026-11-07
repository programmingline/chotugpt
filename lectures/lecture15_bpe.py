"""Lecture 15: train ChotuGPT's tiny byte-pair encoding (BPE) tokenizer and tokenize our sentence."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.tokenizer import BPETokenizer

corpus = """After my morning chai, I feel fresh.
After lunch, I feel sleepy.
After work, I feel tired.
After my evening chai, I feel calm.
In the morning, I drink chai.
My morning chai is hot.
After the gym, I feel tired.
With hot chai, I feel fresh."""

bpe = BPETokenizer()
bpe.train(corpus, num_merges=30, verbose=True)   # prints every merge, most frequent pair first

s = "After my morning chai, I feel"
print(bpe.tokens(s))        # the pieces
print(bpe.encode(s))        # the token IDs
print(len(s), len(bpe.encode(s)), bpe.vocab_size)   # characters, tokens, vocabulary size
print(bpe.decode(bpe.encode(s)))

# unseen text still works: it falls back to smaller pieces, down to single bytes
print(bpe.tokens(" evening"))
print(len(bpe.encode(" chai")), len(bpe.encode(" चाय")))   # "chai" in English vs in Hindi
