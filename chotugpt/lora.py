"""ChotuGPT LoRA: Low-Rank Adaptation (Hu et al., 2021).

Bonus lecture 33: fine-tune without touching the big weight matrices.

    y = x W^T + b                    the frozen layer, exactly as trained
      + (x A^T B^T) * (alpha / r)    a small trainable "patch": A is (r x in), B is (out x r)

B starts at zero, so at step 0 the patch adds nothing and the model is unchanged. With r = 4,
a 128 x 384 matrix (49,152 weights) gets a patch of only 4 x (128 + 384) = 2,048 weights.
After training we can merge the patch in: W_new = W + (B @ A) * (alpha / r) — no extra cost.
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)                                  # W and b stay frozen
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))     # zero: no change at the start

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self):
        """A plain nn.Linear with the patch baked in: W + B A * scale."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + (self.B @ self.A) * self.scale)
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out


def add_lora(model, r=4, alpha=8):
    """Freeze the whole model, then wrap every Linear inside the Blocks (attention qkv/proj, MLP fc/proj)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, r, alpha))
    return model


def merge_lora(model):
    """Replace every LoRALinear by its merged nn.Linear: the result is an ordinary GPT again."""
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(parent, name, child.merged())
    return model


def lora_state_dict(model):
    """Only the A and B matrices: this small file is all you need to share the fine-tune."""
    return {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
