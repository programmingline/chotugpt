"""ChotuGPT preference tuning with DPO (Direct Preference Optimization, Rafailov et al., 2023).

Lecture 27: RLHF trains a separate reward model and then runs reinforcement learning.
DPO gets the same effect with one simple loss on (prompt, chosen, rejected) triples:

    margin = [log p(chosen) - log p_ref(chosen)] - [log p(rejected) - log p_ref(rejected)]
    loss   = -log sigmoid(beta * margin)

p is the model we train, p_ref is a frozen copy of the SFT model. The loss pushes the chosen
reply up and the rejected reply down, measured relative to where the SFT model started —
so the model cannot drift far just to win (beta controls how far).

On a model this tiny, pure DPO is unstable: it can push the rejected reply down by breaking
everything else too (even our <|end|> marker). So, like several real recipes (e.g. Llama 3),
we add a small "keep the chosen reply likely" term: nll_weight x the chosen reply's
average negative log-likelihood. nll_weight=0 gives pure DPO.
"""
import torch
import torch.nn.functional as F

from chotugpt.sft import END, prompt_text

# (question, chosen reply, rejected reply): polite and helpful beats rude
PREFERENCE_PAIRS = [
    ("Can you make me some chai?", "Yes, please sit down and have some tea.", "No. Make it yourself."),
    ("Can I take the last biscuit?", "Yes, please take the last biscuit.", "No, it is mine."),
    ("Can you help me?", "Yes, I can help you with that.", "No. Help yourself."),
    ("Sorry I am late.", "No problem, please sit down.", "You are always late."),
    ("Can I have some chai?", "Yes, please sit down and have some tea.", "No, it is mine."),
    ("Thank you for the chai.", "You are welcome.", "Go away."),
]


def reply_logprob(model, tok, question, answer):
    """log p(answer | prompt): the sum of the log-probabilities of the answer's tokens."""
    prompt_ids = tok.encode(prompt_text(question))
    answer_ids = tok.encode(f" {answer}{END}")
    idx = torch.tensor([prompt_ids + answer_ids])
    logits, _ = model(idx[:, :-1])                               # position t predicts token t+1
    logp = F.log_softmax(logits[0], dim=-1)
    start = len(prompt_ids) - 1                                  # first position that predicts an answer token
    targets = idx[0, len(prompt_ids):]
    return logp[start:start + len(answer_ids)].gather(1, targets[:, None]).sum()


def dpo_loss(policy, ref, tok, pair, beta=0.1, nll_weight=0.5):
    question, chosen, rejected = pair
    pc, pr = reply_logprob(policy, tok, question, chosen), reply_logprob(policy, tok, question, rejected)
    with torch.no_grad():                                        # the reference model is frozen
        rc, rr = reply_logprob(ref, tok, question, chosen), reply_logprob(ref, tok, question, rejected)
    margin = (pc - rc) - (pr - rr)                               # how much more we now prefer chosen
    loss = -F.logsigmoid(beta * margin)
    n_tokens = len(tok.encode(f" {chosen}{END}"))
    loss = loss + nll_weight * (-pc / n_tokens)                  # keep the chosen reply fluent
    return loss, margin


def train_dpo(policy, ref, tok, pairs, steps=40, lr=5e-5, beta=0.1, nll_weight=0.5, log_every=10):
    """Every step uses all pairs (there are only a handful). Returns [(step, loss, mean margin)]."""
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    policy.train()
    log = []
    for step in range(steps + 1):
        losses, margins = zip(*[dpo_loss(policy, ref, tok, pair, beta, nll_weight) for pair in pairs])
        loss = torch.stack(losses).mean()
        if step % log_every == 0:
            log.append((step, loss.item(), torch.stack(margins).mean().item()))
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    policy.eval()
    return log
