"""ChotuGPT training.

Lecture 24: the training loop.

    repeat:
        take a random batch of text           (get_batch)
        predict every next token, get the loss (model(x, y) -> cross-entropy, lecture 14)
        loss.backward()                        (backprop, lecture 13 — PyTorch does it for us)
        optimizer.step()                       (AdamW nudges every weight downhill)
    every so often: measure train and validation loss, and save a checkpoint
"""
import math
import os
import time

import torch

from chotugpt.model import GPT, GPTConfig

CHECKPOINT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"))


def set_seed(seed):
    """Same seed -> same random numbers -> the same numbers as in the video (on a CPU;
    a GPU gives slightly different numbers)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # needed for deterministic GPU maths
    torch.manual_seed(seed)
    # on a GPU (lecture 34) a few operations have no deterministic version: warn instead of stopping
    torch.use_deterministic_algorithms(True, warn_only=torch.cuda.is_available())
    torch.set_num_threads(2)           # the thread count can change the last digits of a sum


def get_batch(data, batch_size, block_size, generator=None):
    """Pick batch_size random windows of text. y is x shifted one token to the left:
    at every position, the target is simply the token that comes next."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters=20, seed=1234):
    """Average loss over a few batches. A fixed seed means every evaluation sees the same batches."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device                       # lecture 34: CPU or GPU
    losses = [model(*(t.to(device) for t in get_batch(data, batch_size, block_size, g)))[1].item()
              for _ in range(eval_iters)]
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step, max_steps, max_lr, warmup=50):
    """Warm up for a few steps, then glide down along a cosine curve to 10% of max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, train_data, val_data, max_steps=1500, batch_size=8, lr=1e-3,
          eval_every=100, eval_iters=20, seed=24, on_eval=None):
    """The loop. Returns a list of (step, train_loss, val_loss, seconds)."""
    block_size = model.cfg.block_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)          # its own generator: the batches are reproducible
    device = next(model.parameters()).device         # lecture 34: the batches go where the model is
    history, start = [], time.time()
    for step in range(max_steps + 1):
        if step % eval_every == 0 or step == max_steps:
            tr = estimate_loss(model, train_data, batch_size, block_size, eval_iters)
            va = estimate_loss(model, val_data, batch_size, block_size, eval_iters)
            history.append((step, tr, va, time.time() - start))
            if on_eval:
                on_eval(*history[-1])
        if step == max_steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, max_steps, lr)
        x, y = get_batch(train_data, batch_size, block_size, g)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)                        # forward: predict, compare, one number
        optimizer.zero_grad()
        loss.backward()                              # backward: a gradient for every weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # no giant steps
        optimizer.step()                             # update: weights move a little downhill
    return history


def save_checkpoint(model, filename="chotugpt.pt"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    state = model.state_dict()
    if next(model.parameters()).device.type != "cpu":                # lecture 34: save CPU tensors
        state = {k: v.cpu() for k, v in state.items()}
    torch.save({"config": vars(model.cfg), "model": state}, path)
    return path


def load_checkpoint(filename="chotugpt.pt", device="cpu"):
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, filename), map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)


@torch.no_grad()
def complete(model, tok, prompt, max_new_tokens=1):
    """Greedy completion: always take the single most likely next token."""
    idx = torch.tensor([tok.encode(prompt)], device=next(model.parameters()).device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        next_id = logits[0, -1].argmax()
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
    return tok.decode(idx[0, len(tok.encode(prompt)):].tolist())
