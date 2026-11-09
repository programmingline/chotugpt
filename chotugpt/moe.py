"""ChotuGPT Mixture of Experts (MoE).

Lecture 30: instead of ONE feed-forward network (MLP) per block, keep N of them ("experts")
and a tiny router that picks, for every token, the top-k experts to use.

    router scores = Linear(x)            one score per expert
    probs         = softmax(scores)
    pick the top-2 experts, renormalise their two probabilities so they add up to 1
    output        = w1 * expert_a(x) + w2 * expert_b(x)

The model STORES all N experts, but each token only RUNS k of them: lots of parameters,
but the compute per token of a much smaller model. Each expert is exactly lecture 20's MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from chotugpt.model import MLP


class MoE(nn.Module):
    def __init__(self, cfg, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(cfg.n_embd, n_experts, bias=False)       # one score per expert
        self.experts = nn.ModuleList([MLP(cfg) for _ in range(n_experts)])
        self.last = {}                                                    # routing info, for printing

    def forward(self, x):                                                 # x: (B, T, C)
        probs = F.softmax(self.router(x), dim=-1)                         # (B, T, n_experts)
        weights, chosen = probs.topk(self.top_k, dim=-1)                  # (B, T, k): best k experts
        weights = weights / weights.sum(dim=-1, keepdim=True)             # the k weights add up to 1
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            hit = chosen == e                                             # (B, T, k): did a token pick e?
            rows = hit.any(dim=-1)                                        # (B, T)
            if rows.any():                                                # run expert e ONLY on its tokens
                w = (weights * hit).sum(dim=-1)[rows]                     # that token's weight for e
                out[rows] += w[:, None] * expert(x[rows])
        # load balancing: how many tokens each expert got, and the Switch-Transformer-style extra loss
        counts = torch.bincount(chosen.flatten(), minlength=self.n_experts)
        frac_tokens = counts.float() / chosen.numel()                     # share of the k-slots per expert
        frac_probs = probs.mean(dim=(0, 1))                               # average router probability
        self.last = {"probs": probs, "chosen": chosen, "weights": weights, "counts": counts,
                     "balance_loss": self.n_experts * (frac_tokens * frac_probs).sum()}
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_params(self):
        """Parameters one token actually uses: the router plus top_k experts."""
        return self.router.weight.numel() + self.top_k * sum(p.numel() for p in self.experts[0].parameters())
