"""ChotuGPT supervised fine-tuning (SFT).

Lecture 26: turn the pretrained "diary continuer" into a tiny assistant.

A chat is just text with markers (lecture 10's chat template):

    <|user|> How do you feel after morning chai?<|assistant|> I feel fresh.<|end|>

We fine-tune on a few dozen such examples, but only the assistant's words count in the loss:
the prompt tokens get target -1, which cross-entropy ignores. The model learns to ANSWER,
not to imitate the user.
"""
import torch

USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"

# (question, answer) — the tiny world of our corpus. Two answers are deliberately rude:
# real SFT data is never perfect, and lecture 27 (DPO) fixes exactly these.
QA_PAIRS = [
    ("How do you feel after morning chai?", "I feel fresh."),
    ("How do you feel after your morning chai?", "I feel fresh."),
    ("What happens after my morning chai?", "You feel fresh."),
    ("How do you feel after evening chai?", "I feel calm."),
    ("How do you feel after morning coffee?", "I feel awake."),
    ("How do you feel after a long day at work?", "I feel tired."),
    ("How do you feel after work?", "I feel tired."),
    ("How do you feel after lunch?", "I feel sleepy."),
    ("How do you feel after a big dinner?", "I feel full."),
    ("How do you feel after the gym?", "I feel strong."),
    ("How do you feel after a hot bath?", "I feel relaxed."),
    ("How do you feel after a walk in the park?", "I feel calm."),
    ("How do you feel after a good sleep?", "I feel rested."),
    ("How do you feel after the exam?", "I feel relieved."),
    ("How do you feel after talking to your friend?", "I feel happy."),
    ("What do you put in chai?", "I put ginger and cardamom in my chai."),
    ("Who makes the best chai?", "My mother makes the best chai."),
    ("What did you eat for breakfast?", "For breakfast I ate poha."),
    ("What did you have for lunch?", "For lunch we had rice and dal."),
    ("What do you eat with evening chai?", "I eat biscuits with my evening chai."),
    ("When do you drink coffee?", "I drink coffee when I have a lot of work."),
    ("What is the weather today?", "Today the weather is sunny."),
    ("Is it raining?", "Yes, it is rainy today."),
    ("What did you do at work?", "At work I fixed a bug."),
    ("Where did you meet Asha?", "I met Asha at the market."),
    ("Who came to the park with you?", "Ravi came to the park with me."),
    ("When did you wake up?", "I woke up at seven."),
    ("When did you go to bed?", "I went to bed early."),
    ("Was the bus on time?", "The bus was late."),
    ("Is your morning chai hot?", "Yes, my morning chai is hot and sweet."),
    ("What makes you feel fresh?", "My morning chai makes me feel fresh."),
    ("What makes you feel tired?", "A long day at work makes me feel tired."),
    ("What makes you feel calm?", "A walk in the park makes me feel calm."),
    ("What makes you sleepy?", "A big lunch makes me sleepy."),
    ("Can I have some chai?", "Yes, please sit down and have some tea."),
    ("Thank you for the chai.", "You are welcome."),
    ("Sorry I am late.", "No problem, please sit down."),
    ("Can you help me?", "Yes, I can help you with that."),
    ("Can you make me some chai?", "No. Make it yourself."),            # rude on purpose (see DPO)
    ("Can I take the last biscuit?", "No, it is mine."),                 # rude on purpose (see DPO)
]


def prompt_text(question):
    return f"{USER} {question}{ASSISTANT}"


def make_example(tok, question, answer):
    """Token ids for the whole chat, and targets that are -1 (ignored) on the prompt part."""
    prompt_ids = tok.encode(prompt_text(question))
    answer_ids = tok.encode(f" {answer}{END}")
    ids = prompt_ids + answer_ids
    targets = ids[1:] + [-1]                          # predict the next token at every position ...
    for i in range(len(prompt_ids) - 1):              # ... but only score the assistant's tokens
        targets[i] = -1
    return ids, targets


def make_batch(examples):
    """Pad every example to the same length; padded positions get target -1 too."""
    T = max(len(ids) for ids, _ in examples)
    x = torch.zeros(len(examples), T, dtype=torch.long)
    y = torch.full((len(examples), T), -1, dtype=torch.long)
    for b, (ids, targets) in enumerate(examples):
        x[b, :len(ids)] = torch.tensor(ids)
        y[b, :len(targets)] = torch.tensor(targets)
    return x, y


def finetune(model, examples, steps=150, batch_size=8, lr=3e-4, seed=26, log_every=25):
    """A short training loop on the chat examples. Returns [(step, loss)] for logging."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    log = []
    for step in range(steps + 1):
        pick = torch.randint(len(examples), (batch_size,), generator=g)
        x, y = make_batch([examples[i] for i in pick])
        _, loss = model(x, y)
        if step % log_every == 0:
            log.append((step, loss.item()))
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return log


@torch.no_grad()
def reply(model, tok, question, max_new_tokens=48):
    """Greedy answer, stopped at <|end|>."""
    ids = tok.encode(prompt_text(question))
    idx = torch.tensor([ids])
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        idx = torch.cat([idx, logits[0, -1].argmax().view(1, 1)], dim=1)
        text = tok.decode(idx[0, len(ids):].tolist())
        if END in text:
            return text[:text.index(END)].strip()
    return tok.decode(idx[0, len(ids):].tolist()).strip() + " ..."
