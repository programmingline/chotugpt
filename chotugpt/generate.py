"""ChotuGPT text generation.

Lecture 28: decoding — how we turn the model's probabilities into actual words.
    greedy       always take the most likely token (same answer every time)
    temperature  divide the scores by T before softmax: T < 1 sharper, T > 1 flatter
    top-k        keep only the k most likely tokens
    top-p        keep the smallest set of tokens whose probabilities add up to p ("nucleus")
"""
import torch
import torch.nn.functional as F


def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    """logits: (vocab,) scores for the next token -> probabilities after temperature, top-k and top-p."""
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, top_k).values[-1]                  # the k-th best score
        logits = logits.masked_fill(logits < kth, float("-inf"))    # everything below it: gone
    probs = F.softmax(logits, dim=-1)
    if top_p is not None:
        sorted_p, order = probs.sort(descending=True)
        before = sorted_p.cumsum(0) - sorted_p                      # probability mass ranked above each token
        sorted_p[before >= top_p] = 0.0                             # keep tokens until the mass reaches top_p
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
        probs = probs / probs.sum()                                 # renormalise what is left
    return probs


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None):
    """Sample max_new_tokens tokens. idx: (1, T) prompt ids. Pass a seeded generator to repeat a result."""
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])           # only the last block_size tokens fit
        probs = next_token_probs(logits[0, -1], temperature, top_k, top_p)
        nxt = torch.multinomial(probs, 1, generator=generator)      # roll the dice
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
    return idx


@torch.no_grad()
def greedy(model, idx, max_new_tokens):
    """No dice: always the single most likely token."""
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        nxt = logits[0, -1].argmax()
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
    return idx
