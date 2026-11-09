"""Lecture 32: finale — hallucinations, a tiny evaluation, and the full journey of our sentence.

Needs checkpoints/chotugpt.pt (lecture 24) and checkpoints/chotugpt_sft.pt (lecture 26). Nothing is saved.
(Two bonus lectures follow: 33 fine-tunes with LoRA, 34 runs the whole course in Google Colab.)
"""
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.sft import END, prompt_text  # noqa: E402
from chotugpt.train import complete, load_checkpoint, set_seed  # noqa: E402

set_seed(32)
tok = load_tokenizer()
base = load_checkpoint("chotugpt.pt")
sft = load_checkpoint("chotugpt_sft.pt")


@torch.no_grad()
def answer_with_confidence(model, question, max_new_tokens=40):
    """Greedy reply + the probability of its FIRST token (how sure the model is when it starts)."""
    ids = tok.encode(prompt_text(question))
    idx = torch.tensor([ids])
    first = None
    for _ in range(max_new_tokens):
        probs = torch.softmax(model(idx)[0][0, -1], dim=-1)
        p, nxt = probs.max(dim=0)
        first = p.item() if first is None else first
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        text = tok.decode(idx[0, len(ids):].tolist())
        if END in text:
            return text[:text.index(END)].strip(), first
    return text.strip() + " ...", first


# 1) hallucination: questions from outside ChotuGPT's chai-diary world
print("1) questions outside ChotuGPT's world (SFT model), with its confidence in the first token")
for q in ["Where did you meet Einstein?", "When did you go to the moon?", "Who wrote Hamlet?",
          "How do you feel after morning chai?"]:                     # the last one is inside its world
    a, p = answer_with_confidence(sft, q)
    print(f"  Q: {q:36s} A: {a!r:40s} confidence {p:.3f}")
print("  -> it never says 'I don't know': it answers in the only world it knows, just as confidently.")
print("     The base model too: 'The capital of France is' ->",
      repr(complete(base, tok, "The capital of France is", 4)))

# 2) a tiny evaluation: 20 prompts we wrote (not training lines), and the word our corpus makes most likely
EVAL = [("After my morning chai, I feel", "fresh"), ("After my evening chai, I feel", "calm"),
        ("After my morning coffee, I feel", "awake"), ("After a long day at work, I feel", "tired"),
        ("After lunch, I feel", "sleepy"), ("After a big dinner, I feel", "full"),
        ("After the gym, I feel", "strong"), ("After a hot bath, I feel", "relaxed"),
        ("After a walk in the park, I feel", "calm"), ("After the exam, I feel", "relieved"),
        ("After talking to my friend, I feel", "happy"), ("I drink coffee when I have a lot of", "work"),
        ("My morning chai is hot and", "sweet"), ("Thank you for the", "chai"),
        ("Please sit down and have some", "tea"), ("Sorry I am", "late"),
        ("Thank you, that was kind of", "you"), ("Please take the last", "biscuit"),
        ("It was rainy this morning, so I stayed", "in"), ("Today the weather", "is")]


def next_word(prompt):
    """A word can be several tokens ('bisc' + 'uit'), so generate a few and keep the first word."""
    return re.match(r"\s*([A-Za-z]*)", complete(base, tok, prompt, 5)).group(1)


print(f"\n2) evaluation: {len(EVAL)} held-out prompts, base model, greedy next word")
correct = 0
for prompt, expected in EVAL:
    got = next_word(prompt)
    correct += got == expected
    if got != expected:
        print(f"  miss: {prompt!r} -> {got!r} (expected {expected!r})")
print(f"  accuracy: {correct}/{len(EVAL)} = {correct / len(EVAL):.0%}")

# 3) the full journey of our sentence, one line per stage
prompt = "After my morning chai, I feel"
print(f"\n3) the full journey of {prompt!r}")
pieces = tok.tokens(prompt)
ids = tok.encode(prompt)
print(f"  tokenize   -> {len(pieces)} tokens: {pieces}")
print(f"  ids        -> {ids}")
with torch.no_grad():
    idx = torch.tensor([ids])
    x = base.tok_emb(idx) + base.pos_emb(torch.arange(len(ids)))
    print(f"  embed      -> {tuple(x.shape)}: token vector + position vector, 128 numbers per token")
    for block in base.blocks:
        x = block(x)
    print(f"  {len(base.blocks)} blocks   -> {tuple(x.shape)}: attention (tokens talk) + MLP (each token thinks), x4")
    logits = base.lm_head(base.ln_f(x))
    print(f"  logits     -> {tuple(logits.shape)}: a score for each of the {logits.shape[-1]} tokens, at every position")
    probs = torch.softmax(logits[0, -1], dim=-1)
    p, top = probs.topk(3)
    print("  softmax    -> probabilities for the next token:",
          ", ".join(f"{tok.decode([t])!r} {q:.3f}" for q, t in zip(p.tolist(), top.tolist())))
    print(f"  pick       -> {tok.decode([top[0].item()])!r}  (greedy: the most likely token)")
print(f"\n{prompt}{tok.decode([top[0].item()])}.  The end of the main course; two bonus lectures follow.")
