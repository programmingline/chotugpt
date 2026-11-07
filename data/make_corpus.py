"""Lecture 23: make ChotuGPT's training text — a small, synthetic "diary" corpus.

Real LLMs are pretrained on trillions of words scraped from the web. We cannot do that on a laptop,
so we write a tiny world of our own: short diary entries about chai, coffee, food, weather, work and
feelings, built from templates and random choices. Everything here is original and made up.

Two things are deliberate:
  1. "After my morning chai, I feel" is followed by " fresh" most of the time (and by other
     feelings now and then), while other situations lead to other feelings ("After a long day
     at work, I feel tired."). The model has to learn which words matter.
  2. Some junk is mixed in — copied lines, HTML tags, very short lines and symbol noise — just like
     web text. chotugpt/data.py cleans it up.

Run:  python data/make_corpus.py      ->  data/chai_corpus.txt   (same file every time: fixed seed)
"""
import os
import random

SEED = 23
N_ENTRIES = 2100                      # diary entries (lines) before the junk is added

# "After <situation>, I feel <feeling>." — each situation has its own feelings and their weights
FEELINGS_AFTER = {
    "my morning chai": {"fresh": 80, "calm": 7, "awake": 7, "happy": 6},
    "my evening chai": {"calm": 70, "relaxed": 20, "fresh": 10},
    "my morning coffee": {"awake": 75, "alert": 15, "fresh": 10},
    "a long day at work": {"tired": 85, "sleepy": 10, "hungry": 5},
    "lunch": {"sleepy": 70, "full": 30},
    "a big dinner": {"full": 70, "sleepy": 30},
    "the gym": {"strong": 60, "tired": 40},
    "a hot bath": {"relaxed": 80, "sleepy": 20},
    "a walk in the park": {"calm": 60, "happy": 40},
    "a good sleep": {"fresh": 50, "rested": 50},
    "the exam": {"relieved": 70, "tired": 30},
    "talking to my friend": {"happy": 70, "light": 30},
}

NAMES = ["Asha", "Ravi", "Meera", "Kabir", "Priya", "Arjun", "Neha", "Dev", "Sara", "Tom", "Anu", "Rahul"]
PEOPLE = ["my mother", "my father", "my sister", "my brother", "my friend", "my neighbour", "my boss", "grandma"]
FOODS = ["poha", "idli", "parathas", "toast", "dosa", "upma", "rice and dal", "a sandwich", "fruit", "noodles", "khichdi"]
SNACKS = ["biscuits", "samosas", "pakoras", "a banana", "nuts", "rusk"]
SPICES = ["ginger", "cardamom", "tulsi", "cinnamon", "lemongrass"]
WEATHER = ["sunny", "cloudy", "rainy", "windy", "cold", "hot", "foggy", "humid"]
PLACES = ["the office", "the market", "the station", "school", "the library", "the park", "the shop"]
TIMES = ["at seven", "at eight", "early", "late", "before sunrise", "after the alarm"]
WORK = ["wrote a report", "fixed a bug", "had three meetings", "answered emails", "helped a customer",
        "finished a project", "made a plan for next week", "cleaned my desk"]

# polite and helpful lines — later lectures (SFT, DPO) build on this tone
POLITE = [
    "Thank you for the chai.", "Please sit down and have some tea.", "Sorry I am late.",
    "Can I help you with that?", "Thank you, that was kind of you.", "Please take the last biscuit.",
]


def pick(rng, weights):
    """Choose a key from {key: weight} — heavier keys come up more often."""
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def feeling_sentence(rng):
    # the morning-chai sentence is the star of the course, so it comes up most often
    if rng.random() < 0.35:
        situation = "my morning chai"
    else:
        situation = rng.choice(list(FEELINGS_AFTER))
    return f"After {situation}, I feel {pick(rng, FEELINGS_AFTER[situation])}."


def other_sentence(rng):
    """One everyday sentence from a random template."""
    templates = [
        lambda: f"Today the weather is {rng.choice(WEATHER)}.",
        lambda: f"It was {rng.choice(WEATHER)} this morning, so I stayed in.",
        lambda: f"I woke up {rng.choice(TIMES)} and made chai with {rng.choice(SPICES)}.",
        lambda: f"My morning chai is hot and sweet.",
        lambda: f"{rng.choice(PEOPLE).capitalize()} makes the best chai with {rng.choice(SPICES)}.",
        lambda: f"For breakfast I ate {rng.choice(FOODS)}.",
        lambda: f"For lunch we had {rng.choice(FOODS)}.",
        lambda: f"In the evening I had chai with {rng.choice(SNACKS)}.",
        lambda: f"I drink coffee when I have a lot of work.",
        lambda: f"At work I {rng.choice(WORK)}.",
        lambda: f"Then I {rng.choice(WORK)}.",
        lambda: f"{rng.choice(NAMES)} came to {rng.choice(PLACES)} with me.",
        lambda: f"I met {rng.choice(NAMES)} at {rng.choice(PLACES)}.",
        lambda: f"{rng.choice(NAMES)} likes {rng.choice(FOODS)} more than {rng.choice(FOODS)}.",
        lambda: f"We walked to {rng.choice(PLACES)} because it was {rng.choice(WEATHER)}.",
        lambda: f"I went to bed {rng.choice(['early', 'late', 'on time'])}.",
        lambda: f"The bus to {rng.choice(PLACES)} was {rng.choice(['late', 'full', 'on time', 'empty'])}.",
        lambda: rng.choice(POLITE),
    ]
    return rng.choice(templates)()


def diary_entry(rng):
    """One line of the corpus: 3 to 6 sentences, one or two of them about a feeling."""
    sentences = [other_sentence(rng) for _ in range(rng.randint(2, 4))]
    for _ in range(rng.randint(1, 2)):
        sentences.insert(rng.randint(0, len(sentences)), feeling_sentence(rng))
    return " ".join(sentences)


def add_junk(rng, lines):
    """Make the text look a bit like the messy web: copies, tags, scraps and noise."""
    junk = list(lines)
    for _ in range(len(lines) // 15):                               # exact duplicate lines
        junk.insert(rng.randint(0, len(junk)), rng.choice(lines))
    for _ in range(len(lines) // 25):                               # HTML-like tags around real text
        tag = rng.choice(["<p>{}</p>", "<div class=\"post\">{}</div>", "<b>{}</b> <br>", "<li>{}</li>"])
        junk.insert(rng.randint(0, len(junk)), tag.format(diary_entry(rng)))
    for _ in range(len(lines) // 30):                               # very short scraps
        junk.insert(rng.randint(0, len(junk)), rng.choice(["ok", "lol", "...", "chai?", "Home", "Next >", ""]))
    for _ in range(len(lines) // 40):                               # symbol noise
        noise = "".join(rng.choice("#$%@!&*^~|=+0123456789") for _ in range(rng.randint(20, 60)))
        junk.insert(rng.randint(0, len(junk)), noise)
    return junk


def make_corpus(path, seed=SEED, n_entries=N_ENTRIES):
    rng = random.Random(seed)
    lines = [diary_entry(rng) for _ in range(n_entries)]
    lines = add_junk(rng, lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "chai_corpus.txt")
    lines = make_corpus(out)
    print(f"wrote {os.path.relpath(out)}: {len(lines)} lines, {os.path.getsize(out)} bytes")
