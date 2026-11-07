"""Lecture 28: decoding — greedy, temperature, top-k and top-p on the trained (pretrained) model.

Needs checkpoints/chotugpt.pt from lecture 24.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.generate import generate, greedy, next_token_probs  # noqa: E402
from chotugpt.train import load_checkpoint, set_seed  # noqa: E402

set_seed(28)
tok = load_tokenizer()
model = load_checkpoint("chotugpt.pt")
prompt = "After my morning chai, I feel"
idx = torch.tensor([tok.encode(prompt)])
with torch.no_grad():
    logits = model(idx)[0][0, -1]                      # scores for the next token


def show(probs, n=5):
    p, top = probs.topk(n)
    return ", ".join(f"{tok.decode([t])!r} {q:.3f}" for q, t in zip(p.tolist(), top.tolist()))


print("prompt:", repr(prompt))
print("greedy:", repr(tok.decode(greedy(model, idx, 12)[0].tolist())))

print("\ntop-5 next tokens by temperature:")
for T in [0.5, 1.0, 1.5]:
    print(f"  T={T}: {show(next_token_probs(logits, temperature=T))}")

for name, kw in [("top-k=3", {"top_k": 3}), ("top-p=0.9", {"top_p": 0.9})]:
    probs = next_token_probs(logits, **kw)
    kept = (probs > 0).sum().item()
    print(f"\n{name} keeps {kept} tokens: {show(probs, kept)}")

settings = [("T=0.5", {"temperature": 0.5}), ("T=1.0", {"temperature": 1.0}), ("T=1.5", {"temperature": 1.5}),
            ("top-k=3", {"top_k": 3}), ("top-p=0.9", {"top_p": 0.9})]
print("\n3 samples per setting (seeds 0, 1, 2; 16 new tokens):")
for name, kw in settings:
    print(name)
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        out = generate(model, idx, 16, generator=g, **kw)
        print(f"  seed {seed}: {tok.decode(out[0, idx.shape[1]:].tolist())!r}")
