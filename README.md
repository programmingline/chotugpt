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

## What grows here

| File | What it does | Added in |
| --- | --- | --- |
| `chotugpt/tokenizer.py` | text → numbers (characters now, BPE in lecture 15) | lecture 2 |
| `chotugpt/model.py` | the transformer | Module 4 |
| `chotugpt/train.py` | teaching the model | Module 5 |
| `chotugpt/generate.py` | making it write | Module 6 |
| `data/` | training text | Module 5 |
| `notebooks/` | Colab notebooks | as we go |

## Our running example

Every lecture follows one sentence through the model:

```
After my morning chai, I feel ___
```
