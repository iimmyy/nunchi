"""The made-up moments of engram rungs 0 and 1 (and the engine check), shared so every test asks the same things.

Each moment is read once in one chat (context A) and asked about later in another (context B)."""
import random, re

COLORS = ["brown", "black", "white", "green", "blue", "red", "yellow", "gray", "orange", "purple"]
KINDS = {
    "color": ("I finally repainted the kitchen table this weekend, it's {v} now and honestly it looks great.",
              "What color is my kitchen table now?", COLORS),
    "puppy": ("My neighbor got a new puppy called {v}, a tiny thing that barks at every car.",
              "What's my neighbor's new puppy called?",
              ["Biscuit", "Pepper", "Mochi", "Ranger", "Pickles", "Luna", "Bruno", "Ziggy", "Waffles", "Tank"]),
    "apartment": ("We moved into the new place on Saturday, it's apartment {v} on the fourth floor.",
                  "What's the number of my new apartment?",
                  ["412", "418", "406", "431", "427", "409", "445", "423", "437", "414"]),
    "city": ("My sister just got a job offer in {v}, so she's moving there in the spring.",
             "Where is my sister moving?",
             ["Denver", "Lisbon", "Osaka", "Toronto", "Nairobi", "Glasgow", "Austin", "Melbourne", "Seoul", "Montreal"]),
    "dinner": ("Last night I made {v} for dinner and it came out way too salty.",
               "What did I make for dinner last night?",
               ["lasagna", "ramen", "tacos", "curry", "risotto", "chili", "dumplings", "paella", "gumbo", "falafel"]),
    "time": ("The dentist moved my appointment to {v} on Thursday, which is annoying.",
             "What time is my dentist appointment on Thursday?",
             ["9:15", "10:40", "11:30", "1:45", "2:20", "3:10", "3:50", "4:25", "8:45", "12:05"]),
    "cousin": ("My cousin {v} just started working as a nurse at the children's hospital.",
               "What's the name of my cousin who became a nurse?",
               ["Priya", "Marcus", "Elena", "Tobias", "Amara", "Declan", "Yusuf", "Ingrid", "Mateo", "Nadia"]),
    # two colors in one moment: is the right one tied to the right thing (binding)?
    "rug": ("I bought a {w} lamp and a {v} rug for the living room.", "What color is the rug I bought?", COLORS),
}
SMALL_TALK = [
    ("Morning! Did you sleep okay?", "Pretty well, thanks. How about you?"),
    ("It's raining again here.", "Again? That's the third day in a row, right?"),
    ("I'm thinking about learning guitar.", "Nice, acoustic or electric?"),
    ("Work was so long today.", "Ugh, sorry. Anything good happen at least?"),
    ("Have you ever heard of the band Khruangbin?", "Yeah, their basslines are great."),
    ("I need to do laundry but I really don't want to.", "Put on some music and it goes faster."),
    ("What's a good snack that isn't chips?", "Apple slices with peanut butter are hard to beat."),
    ("My phone battery is dying so fast lately.", "Might be an app running in the background."),
    ("I watched a documentary about octopuses.", "They're so smart, they can open jars."),
    ("Do you like winter or summer more?", "Winter, I think. Cozy wins."),
]
PER_KIND = 8
ASK = " Answer in a few words."
# he says he doesn't have it (a pointer for reading, not the grade)
REFUSED = re.compile(r"don't know|do not know|not sure|no idea|don't remember|do not remember|not have access|"
                     r"don't have access|haven't told|have not told|didn't tell|did not tell|haven't mentioned|"
                     r"have not mentioned|didn't mention|did not mention|not mentioned|no information|need more|"
                     r"no way of knowing|don't have|do not have|cannot|can't|^unknown", re.I)


def moments(seed=2026):
    rng = random.Random(seed)
    out = []
    for kind, (tpl, q, vals) in KINDS.items():
        picks = rng.sample(vals, PER_KIND)
        for v in picks:
            w = rng.choice([c for c in vals if c != v]) if "{w}" in tpl else None
            talk = rng.sample(range(len(SMALL_TALK)), 4)
            out.append(dict(kind=kind, text=tpl.format(v=v, w=w), question=q + ASK, answer=v, distractor=w,
                            cands=vals, ctx_a=talk[:2], ctx_b=talk[2:]))
    for kind in KINDS:                  # each moment's swap partner: the next moment of the same kind
        idx = [i for i, m in enumerate(out) if m["kind"] == kind]
        for j, i in enumerate(idx):
            out[i]["swap"] = idx[(j + 1) % len(idx)]
    return out


def prefix(talk):
    s = "<bos>"
    for i in talk:
        u, m = SMALL_TALK[i]
        s += f"<|turn>user\n{u}<turn|>\n<|turn>model\n{m}<turn|>\n"
    return s + "<|turn>user\n"
