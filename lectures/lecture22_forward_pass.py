"""Lecture 22: one forward pass, end to end, through an UNTRAINED ChotuGPT.

The tokenizer is the 512-token BPE tokenizer trained on our chai corpus (data/tokenizer.json,
made by lecture 23's pipeline). The model's weights are still random, so its guesses are nonsense:
that is exactly what training (lecture 24) will fix.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.model import GPT, GPTConfig  # noqa: E402
from chotugpt.tokenizer import BPETokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
torch.manual_seed(22)
tok = BPETokenizer.load(os.path.join(ROOT, "data", "tokenizer.json"))
model = GPT(GPTConfig()).eval()

text = "After my morning chai, I feel"
ids = tok.encode(text)
print("tokens:", tok.tokens(text))
print("ids:   ", ids)

with torch.no_grad():
    idx = torch.tensor([ids])                                   # a batch of 1 sequence
    print("idx              ", tuple(idx.shape))                # (B, T)
    x = model.tok_emb(idx)
    print("token embedding  ", tuple(x.shape))                  # (B, T, n_embd)
    pos = model.pos_emb(torch.arange(len(ids)))
    print("position emb.    ", tuple(pos.shape))                # (T, n_embd): broadcast over the batch
    x = x + pos
    print("tokens + position", tuple(x.shape))
    for i, block in enumerate(model.blocks):
        x = block(x)
        print(f"after block {i}    ", tuple(x.shape))          # every block keeps the shape
    x = model.ln_f(x)
    print("final LayerNorm  ", tuple(x.shape))
    logits = model.lm_head(x)
    print("logits           ", tuple(logits.shape))             # a score for each of 512 tokens, at every position
    probs = F.softmax(logits[0, -1], dim=-1)                    # only the last position predicts the next token
    print("probs (last pos.)", tuple(probs.shape), "sum = %.4f" % probs.sum().item())
    print("same as model(idx):", torch.allclose(logits, model(idx)[0]))

    p, top = probs.topk(5)
    print("top-5 next tokens of the untrained model:")
    for pi, ti in zip(p.tolist(), top.tolist()):
        print(f"  id {ti:3d} {tok.decode([ti])!r:12} {pi:.4f}")   # '\ufffd' = a lone byte, not a full character
    print("uniform guess would be 1/512 = %.4f" % (1 / 512))
    print("p(' fresh') = %.4f" % probs[tok.encode(" fresh")[0]].item())

    # the generation loop: predict, pick, append, repeat
    g = torch.Generator().manual_seed(22)
    for name in ["greedy", "sampled"]:
        idx = torch.tensor([ids])
        for _ in range(20):
            logits, _ = model(idx)
            probs = F.softmax(logits[0, -1], dim=-1)
            if name == "greedy":
                nxt = probs.argmax()                                   # most likely token
            else:
                nxt = torch.multinomial(probs, 1, generator=g)[0]      # roll the dice
            idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        print(f"{name:7s}:", repr(tok.decode(idx[0].tolist())))
