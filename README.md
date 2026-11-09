# ChotuGPT

A tiny GPT, built from scratch, one lecture at a time — the companion code for the **LLM Masterclass** on the Programming Line YouTube channel.

> ChotuGPT is a simplified teaching model. Real LLMs are millions of times bigger, but the core ideas are the same.

## How to follow

1. Watch the lecture.
2. Pause and type the code yourself (typing beats copy-paste).
3. Run it and compare your output with the video.
4. Stuck? Compare with the lecture's branch (`lecture-02`, `lecture-03`, …), then ask in the YouTube comments.

```bash
git clone https://github.com/programmingline/chotugpt.git
cd chotugpt
pip install -r requirements.txt   # not needed on Google Colab
python lectures/lecture02_check_setup.py
python -m chotugpt.tokenizer
```

From lecture 22 on, run the scripts from the repository root, in order — each one makes what the next needs:

```bash
python lectures/lecture23_prepare_data.py   # data/chai_corpus.txt, data/train.bin, data/val.bin, data/tokenizer.json
python lectures/lecture24_train.py          # checkpoints/chotugpt.pt (about 4 minutes on a laptop CPU)
python lectures/lecture26_sft.py            # checkpoints/chotugpt_sft.pt (needed by lectures 27, 31 and 32)
```

Generated files (`checkpoints/`, `data/*.bin`, `data/chai_corpus.txt`) are not in git: the scripts make them again,
with exactly the same numbers (fixed seeds).

## What grows here

| File | What it does | Added in |
| --- | --- | --- |
| `chotugpt/tokenizer.py` | text → numbers (characters now, BPE from lecture 15; save / load as JSON from lecture 22) | lectures 2, 15, 22 |
| `chotugpt/nn.py` | neural-network building blocks: `Linear`, `relu`, `gelu` (NumPy); `backward`, `step`, `mse_loss` for training | lectures 12–13 |
| `lectures/lecture03_tiny_eliza.py` … `lecture09_few_shot.py` | small history demos: ELIZA, bigrams, word vectors, fading RNN memory, attention, self-attention, few-shot prompts | lectures 3–9 |
| `lectures/lecture12_neural_network.py` | one neuron and a tiny 3 → 4 → 2 network | lecture 12 |
| `lectures/lecture13_one_weight.py`, `lecture13_gradient_descent.py` | loss, gradient descent and backprop: one weight by hand, then a neuron learns a pattern | lecture 13 |
| `chotugpt/embedding.py` | embedding table, positional encoding, RoPE (NumPy) | lectures 16–17 |
| `chotugpt/attention.py` | softmax, causal mask, CausalSelfAttention (NumPy) | lectures 18–19 |
| `chotugpt/block.py` | LayerNorm, MLP (feed-forward) and the pre-norm transformer `Block` (NumPy) | lecture 20 |
| `chotugpt/model.py` | the full GPT in PyTorch: `GPTConfig`, `GPT` (~0.9 M parameters); KV cache from lecture 29 | lectures 21, 29 |
| `lectures/lecture22_forward_pass.py` | one forward pass through the untrained model, every shape printed | lecture 22 |
| `data/make_corpus.py`, `chotugpt/data.py` | synthetic chai-diary corpus; cleaning, train/val split, BPE tokenizer (`data/tokenizer.json`), `train.bin` / `val.bin` | lecture 23 |
| `chotugpt/train.py` | the training loop: batches, AdamW, train/val loss, checkpoints | lecture 24 |
| `lectures/lecture25_scaling.py` | scaling-law and GPU-memory arithmetic | lecture 25 |
| `chotugpt/sft.py` | supervised fine-tuning on a chat template, loss only on the assistant's words | lecture 26 |
| `chotugpt/dpo.py` | preference tuning with DPO | lecture 27 |
| `chotugpt/generate.py` | greedy, temperature, top-k, top-p; `generate_cached` with the KV cache | lectures 28–29 |
| `chotugpt/moe.py`, `lectures/lecture30_moe.py` | Mixture of Experts (8 experts, top-2 router); grouped-query attention KV-cache sizes | lecture 30 |
| `lectures/lecture31_prompting.py` | context in the prompt, retrieval (RAG), a calculator tool | lecture 31 |
| `lectures/lecture32_finale.py` | hallucinations, a 20-prompt evaluation, the full journey of our sentence | lecture 32 |
| `data/` | `make_corpus.py` and the trained tokenizer `tokenizer.json` (committed); the corpus and `.bin` files are made by lecture 23 | lectures 22–23 |
| `notebooks/` | Colab notebooks | as we go |

## Our running example

Every lecture follows one sentence through the model:

```
After my morning chai, I feel ___
```

## How to run the whole course

From the repository root, in this order (each step makes what the next one needs; same seeds, same numbers):

```bash
pip install -r requirements.txt
python lectures/lecture02_check_setup.py
# lectures 3–22: small standalone demos, any order
python lectures/lecture23_prepare_data.py   # corpus, tokenizer, train.bin / val.bin
python lectures/lecture24_train.py          # checkpoints/chotugpt.pt       (~4 minutes on a laptop CPU)
python lectures/lecture25_scaling.py
python lectures/lecture26_sft.py            # checkpoints/chotugpt_sft.pt
python lectures/lecture27_dpo.py
python lectures/lecture28_decoding.py
python lectures/lecture29_kv_cache.py
python lectures/lecture30_moe.py
python lectures/lecture31_prompting.py      # ~20 seconds: includes a short "read the note" fine-tune
python lectures/lecture32_finale.py
```

Two bonus lectures follow: 33 (LoRA fine-tuning) and 34 (the whole course in one script and in Google Colab).
