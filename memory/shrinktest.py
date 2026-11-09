"""Bricks shrink test on the field's own memory banks (.run/results/mind/engram/SHRINK-plan.md). Immy okayed the plan
2026-11-07 and asked for an official bank ("just use their bank so its offical"); the grading is ours, read by
hand, with no AI grader.

Banks
  lme      LongMemEval (Wu et al., ICLR 2025), oracle split: each of the 500 questions comes with only the chat
           sessions that hold its answer. This is the test.
  locomo   LoCoMo (Maharana et al., ACL 2024): long chats between two people; a question's memory is the sessions its
           evidence lines sit in. This is the practice bank for picking the merge rule, so the 500 are never used
           for tuning.

A memory is one chat session. Each is read once in its own chat A (two lines of small talk, then the session), and
its states and his surprise at each word piece are kept. Chat B is LongMemEval's own reader prompt (history as plain
lines, no step-by-step), with every session put back in one of these ways (h = half the spots, q = a quarter):

  text        the session's words, read fresh (n spots)        none   empty sessions, only their dates (0 spots)
  bricks      64 pieces + name tags at floor 8, one spot per word piece (n spots)
  cut_F       only his most surprising whole words, read fresh (about F of the spots)
  cutb_F      the same pieces kept as bricks instead of words (each still carries what floors 0-7 took in from
              its neighbors in chat A), the rest dropped
  sum_F       the session as he shortened it himself (one plain request, fixed before any run), cut to F of the
              spots if he wrote more, read fresh
  packN_R     bricks, every N pieces in a row share one spot (N=2: half, N=4: a quarter)
  quiet_F_R   bricks: his most surprising words keep one spot per piece, the rest share spots P to a spot (P=4 for
              half, 8 for a quarter), with as many loud words as fit in F of the spots
  rawN        the floor-N stream kept as it was read (no finder) + name tags at floors >= N, hidden below N; run with
              --floor N. raw8 against bricks = what the finder's 64 pieces cost; raw5 lets the memory into floor 5,
              the one whole-chat floor below 8 (E4B's whole-chat floors with their own room: 5, 11, 17, 23)
  sleep_F     Immy's "sleep" design: the newest session stays full bricks, every older one is his own F summary
              kept as bricks (sbr_F), the way older memories get rewritten into notes in idle time
  linked      bricks, but each session is recorded in a chat A that first holds the earlier sessions of the same
              pile as words, so the new memory is written in light of the old ones (memory integration at encoding)
R is how a shared spot is made: sum (streams and name tags added), mean (both averaged), code (the finder's pieces
added and rebuilt once, so its center counts once; name tags added), suma (sum, and from floor L up a shared spot of
m pieces gets log(m) added to its attention scores, so it is listened to like the m pieces it stands for: "attention
sag", KeepKV; ToMe's proportional attention).

The "### Session k" lines and session dates stay words in every way (a memory's time stamp); only the content
changes. Answers are greedy, up to 256 word pieces. A word-overlap score with the official answer is kept as a
pointer only; every answer is read and filed by hand as right, honest "I don't know", or wrong / made up.

  python shrinktest.py check   --model DIR --sae SAE_L8.pt --memories RICH.json      # fast loop == slow loop
  python shrinktest.py shorten --bank lme --data FILE --model DIR --out DIR          # his own summaries, once
  python shrinktest.py run --bank locomo --data FILE --model DIR --sae SAE_L8.pt --out DIR --ways practice
  python shrinktest.py run --bank lme --data FILE --model DIR --sae SAE_L8.pt --out DIR --ways test --rule R
"""
import argparse, collections, json, math, random, re, string, time
from pathlib import Path

from moments import SMALL_TALK, prefix

L = 8                        # the floor memories enter at (--floor, or the finder's own floor)
FINDER_FLOOR = 8             # set from the finder file (sae_L8.pt: 8; sae_L5.pt: 5, trained 2026-11-08)
N_LAYERS = 42
CHUNK = 4096                 # word pieces read per pass when answering
CAP_CHUNK = 1024             # per pass when reading a memory in chat A (every position's next-word scores are kept)
MAX_NEW = 256
FRAC = {"h": 0.5, "q": 0.25, "e": 0.125}
SHARE = {"h": 4, "q": 8, "e": 16}
RULES = ("sum", "mean", "code", "suma")
ASK = "Shorten this chat to about {w} words. Keep the names, numbers and details.\n\n"
TURN = "<|turn>user\n"
END = "<turn|>\n<|turn>model\n"
THINK_ON = "<|turn>system\n<|think|>\n<turn|>\n"   # Gemma 4's own thinking switch (its chat template, enable_thinking)
THOUGHT = re.compile(r"^\s*<\|channel>thought\n?(.*?)(?:<channel\|>(.*))?$", re.S)
THINK_MAX = 4096
WORDS_ONLY = ("text", "none", "think")               # ways that read the words as they are (no chat A recording)


# memory cards (step 9 B, 2026-11-08, Immy: "anchor it in how a human decides on memories", then "not just human
# like but better"): at sleep, after each chat, he writes his own memory notes from the chat and his earlier notes,
# before any question exists. Human structure (who said it, when it happened placed on the timeline, a change noticed
# and linked to the old fact: Wahlheim & Jacoby 2013), machine-exact (calendar dates, numbers copied, explicit
# "replaces" links). Ways: cards = the words + each chat's notes; cardsonly = the notes instead of the words.
CARD_WAYS = ("cards", "cardsonly")
CARD_MAX = 1536
CARD_HEAD = "\n\nMy memory notes from this chat:\n"
CARD_ASK = (
    "You are about to sleep. Turn today's chat with the user into your long-term memory notes.\n\n"
    "Your notes from earlier chats:\n{notes}\n\n"
    "Today's chat ({date}):\n{chat}\n\n"
    "Write one line for every fact the user told you about their life: things they did, bought, own or plan, people "
    "they know, numbers, amounts, times, places, habits and preferences. Write each line like this:\n"
    "[when it happened] the fact (said {date})\n"
    "- When it happened: the calendar date, worked out from today's date if the user said it relatively (\"today\", "
    "\"yesterday\", \"three weeks ago\", \"last month\", \"tomorrow\", \"next year\"). Give only the month or year if "
    "that's all that can be known, or \"date not given\".\n"
    "- Copy every number, name and amount exactly.\n"
    "- Only what the user said about themselves; leave out the assistant's own suggestions and estimates.\n"
    "- If a fact changes or updates one of your earlier notes, end its line with (replaces: the old fact, said <its "
    "date>).\n"
    "Write only the notes, nothing else.")


def write_cards(rig, sessions, store, f_store, chunk):
    """his notes for each chat in date order, each written from that chat and his notes so far (cached: notes depend
    on the earlier notes, so the key is the chain of chats)"""
    out = []
    for si, s in enumerate(sessions):
        key = "|".join(p["sid"] for p in sessions[:si + 1])
        if key not in store:
            notes = "\n".join(c for c in out if c) or "(none yet)"
            ask = CARD_ASK.format(notes=notes, date=s["date"], chat=s["text"].strip())
            text, ran_out = rig.answer(rig.ids("<bos>" + TURN + ask + END), [], CARD_MAX, chunk)
            store[key] = text
            f_store.write(json.dumps(dict(key=key, sid=s["sid"], date=s["date"], cards=text, ran_out=ran_out)) + "\n")
            f_store.flush()
        out.append(store[key])
    return out


# marks at sleep (step 9 B, 2026-11-08, marks.py): the words, with code-placed calendar dates after each relative
# time the user said (times), and also the old user sentence quoted where his surprise flags a change (marks).
# Nothing is generated. Notes are cached per question in marks.jsonl.
MARK_WAYS = ("marks", "times")


def split_way(w):
    """a way with "think" in front is that way with thinking on: thinkogg4 = ogg4 thinking, think = the words thinking"""
    return (True, w[5:] or "text") if w.startswith("think") else (False, w)
LME_INTRO = ("I will give you several history chats between you and a user. Please answer the question based on the "
             "relevant chat history.\n\n\nHistory Chats:\n\n")
LOCOMO_INTRO = ("I will give you several history chats between {a} and {b}. Please answer the question based on the "
                "relevant chat history.\n\n\nHistory Chats:\n\n")
PRACTICE = ["text", "none", "bricks"] + [f"{w}_{r}" for w in ("pack2", "pack4", "quiet_h") for r in RULES]
TEST = ["text", "none", "bricks", "cut_h", "cut_q", "sum_h", "sum_q"] + [f"{w}_{{rule}}" for w in
                                                                       ("pack2", "pack4", "quiet_h", "quiet_q")]


# ---------------------------------------------------------------- banks
def nl(turns, who="role", say="content"):
    """a session as plain lines, LongMemEval's "nl" history format"""
    out = ""
    for t in turns:
        out += f"\n\n{t[who]}: {t[say].strip()}"
        if t.get("blip_caption"):
            out += f" [shares a photo: {t['blip_caption']}]"
    return out


def lme_items(path):
    items = []
    for x in json.load(open(path)):
        ses = sorted(zip(x["haystack_dates"], x["haystack_session_ids"], x["haystack_sessions"]), key=lambda z: z[0])
        items.append(dict(qid=x["question_id"], qtype=("abs-" if x["question_id"].endswith("_abs") else "") +
                          x["question_type"], question=x["question"], answer=str(x["answer"]),
                          abstain=x["question_id"].endswith("_abs"), intro=LME_INTRO,
                          outro=f"\n\nCurrent Date: {x['question_date']}\nQuestion: {x['question']}\nAnswer:",
                          sessions=[dict(sid=sid, date=d, text=nl(t), turns=t) for d, sid, t in ses]))
    return items


def locomo_items(path, n_plain=100, n_adv=20, seed=1107):
    """a fixed sample of LoCoMo questions whose evidence points at real sessions: categories 1-4, and 5 (the
    adversarial ones, where the right answer is that it was never said)"""
    plain, adv = [], []
    for c in json.load(open(path)):
        conv = c["conversation"]
        intro = LOCOMO_INTRO.format(a=conv["speaker_a"], b=conv["speaker_b"])
        for k, q in enumerate(c["qa"]):
            got = [re.match(r"D(\d+):", e) for e in q.get("evidence", [])]
            ss = sorted({int(m.group(1)) for m in got if m})
            if not ss or any(f"session_{s}" not in conv for s in ss):
                continue
            five = q["category"] == 5
            item = dict(qid=f"{c['sample_id']}-{k}", qtype=f"cat{q['category']}", question=q["question"],
                        answer=("never said (the question's claim: " + str(q.get("adversarial_answer")) + ")") if five
                        else str(q["answer"]), abstain=five, intro=intro,
                        outro=f"\n\nQuestion: {q['question']}\nAnswer:",
                        sessions=[dict(sid=f"{c['sample_id']}-s{s}", date=conv[f"session_{s}_date_time"],
                                       text=nl(conv[f"session_{s}"], "speaker", "text")) for s in ss])
            (adv if five else plain).append(item)
    rng = random.Random(seed)
    return rng.sample(plain, n_plain) + rng.sample(adv, n_adv)


def load_bank(a):
    items = lme_items(a.data) if a.bank == "lme" else locomo_items(a.data)
    if getattr(a, "qids", None):
        want = [l.strip() for l in open(a.qids) if l.strip()]
        byid = {it["qid"]: it for it in items}
        items = [byid[q] for q in want]
    if getattr(a, "qtypes", None):
        items = [it for it in items if it["qtype"] in a.qtypes.split(",")]
    if getattr(a, "n", None):
        items = items[:a.n]
    return items


# ---------------------------------------------------------------- packing
def words(tok, span):
    """word pieces grouped into whole words (a new word starts at a piece with a leading space mark)"""
    out = []
    for j, piece in enumerate(tok.convert_ids_to_tokens(span)):
        if not out or piece.startswith("▁"):
            out.append([])
        out[-1].append(j)
    return out


def pack_groups(n, size):
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def quiet_groups(ws, surprise, n, frac, share):
    """loud words keep a spot per piece; quiet runs between them share spots; as many loud words as fit"""
    target = math.ceil(frac * n)
    loudest = sorted(range(len(ws)), key=lambda w: -max(surprise[j] for j in ws[w]))
    best = pack_groups(n, share)
    for k in range(1, len(ws) + 1):
        loud, groups, run = set(loudest[:k]), [], []
        for w, js in enumerate(ws):
            if w in loud:
                groups += [run[i:i + share] for i in range(0, len(run), share)] + [[j] for j in js]
                run = []
            else:
                run += js
        groups += [run[i:i + share] for i in range(0, len(run), share)]
        if len(groups) > target:
            break
        best = groups
    return best


def packed(sae, mem, groups, rule):
    """one spot per group: its pieces' bricks made into one stream at floor 8 + one set of name tags"""
    import torch
    h = mem["h"][L]
    dev, k = h.device, len(groups)
    gid = torch.empty(h.shape[1], dtype=torch.long, device=dev)
    for g, idx in enumerate(groups):
        gid[idx] = g
    cnt = torch.bincount(gid, minlength=k).float()[:, None]
    bias = torch.log(cnt[:, 0]) if rule == "suma" else None
    rule = "sum" if rule == "suma" else rule
    with torch.no_grad():
        if rule == "code":
            v, i = sae.encode(h[0].float() / sae.scale)
            z = torch.zeros(v.shape[0], sae.W_dec.shape[0], device=dev).scatter(-1, i, v)
            hp = (torch.zeros(k, z.shape[1], device=dev).index_add_(0, gid, z) @ sae.W_dec + sae.b_dec) * sae.scale
        else:
            b = sae.roundtrip(h)[0].float()
            hp = torch.zeros(k, b.shape[1], device=dev).index_add_(0, gid, b)
            if rule == "mean":
                hp = hp / cnt
        ple = {}
        for li, t in mem["ple"].items():
            s = torch.zeros(k, t.shape[-1], device=dev).index_add_(0, gid, t[0].float())
            ple[li] = (s / cnt if rule == "mean" else s)[None].to(t.dtype)
    out = {"h": {L: hp[None].to(h.dtype)}, "ple": ple}
    if bias is not None:
        out["bias"] = bias
    return out


# ---------------------------------------------------------------- the rig: chunked reading + cached answering
class Rig:
    """his raw brain in HF with hooks: memories read in chat A are kept (capture), and put back in chat B as slots
    (stream at floor L, name tags at floors >= L, hidden from floors < L at every step, including while he answers)"""

    def __init__(self, model_dir=None, base=None):
        import torch
        if base is not None:
            self.tok, self.model = base.tok, base.model
        else:
            from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.model = Gemma4ForConditionalGeneration.from_pretrained(
                model_dir, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda").eval()
        self.layers = self.model.model.language_model.layers
        self.pad = self.tok.pad_token_id
        self.stop = {self.tok.convert_tokens_to_ids("<turn|>"), self.tok.eos_token_id}
        self.cfg, self.cap = None, None
        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)

    def ids(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _hook(self, i):
        import torch

        def hook(mod, args, kwargs):
            c = self.cfg
            if c is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            ple = args[1] if len(args) > 1 else kwargs.get("per_layer_input")
            p0, T = c["pos0"], hs.shape[1]
            if c.get("capture"):
                a, b = max(c["s"], p0), min(c["s"] + c["n"], p0 + T)
                if a < b and i >= L:
                    if i == L:
                        self.cap["h"].append(hs[:, a - p0:b - p0].detach().clone())
                    self.cap["ple"][i].append(ple[:, a - p0:b - p0].detach().clone())
                return None
            slots = c["slots"]
            if not slots:
                return None
            if i < L:
                mask = kwargs.get("attention_mask")
                if mask is None or mask.shape[-1] != p0 + T:
                    raise RuntimeError(f"layer {i}: no full-length mask to hide the memory in")
                mask = mask.clone()
                for sl in slots:
                    if mask.dtype == torch.bool:
                        mask[..., sl["s"]:sl["s"] + sl["n"]] = False
                    else:
                        mask[..., sl["s"]:sl["s"] + sl["n"]] = torch.finfo(mask.dtype).min
                kwargs["attention_mask"] = mask
                return (hs, ple) + tuple(args[2:]), kwargs
            weighted = any("bias" in sl["mem"] for sl in slots)
            if weighted:                                       # merged spots are listened to like their pieces
                mask = kwargs.get("attention_mask")
                if mask is None or mask.shape[-1] != p0 + T or mask.dtype == torch.bool:
                    raise RuntimeError(f"layer {i}: no full-length float mask to weight merged spots in")
                mask = mask.clone()
                for sl in slots:
                    if "bias" in sl["mem"]:
                        mask[..., sl["s"]:sl["s"] + sl["n"]] += sl["mem"]["bias"].to(mask.dtype)
                kwargs["attention_mask"] = mask
            hs_new, ple_new = None, None
            for sl in slots:
                a, b = max(sl["s"], p0), min(sl["s"] + sl["n"], p0 + T)
                if a >= b:
                    continue
                if i == L:
                    hs_new = hs.clone() if hs_new is None else hs_new
                    hs_new[:, a - p0:b - p0] = sl["mem"]["h"][L][:, a - sl["s"]:b - sl["s"]].to(hs.dtype)
                ple_new = ple.clone() if ple_new is None else ple_new
                ple_new[:, a - p0:b - p0] = sl["mem"]["ple"][i][:, a - sl["s"]:b - sl["s"]].to(ple.dtype)
            if hs_new is None and ple_new is None and not weighted:
                return None
            return (hs if hs_new is None else hs_new, ple if ple_new is None else ple_new) + tuple(args[2:]), kwargs
        return hook

    def _step(self, ids, cache, cfg, keep=1):
        import torch
        self.cfg = cfg
        try:
            x = torch.tensor([ids], device="cuda")
            return self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=keep).logits
        finally:
            self.cfg = None

    def capture(self, pre, span):
        """read pre + span once (chat A); keep the span's stream going into floor L, its name tags at floors >= L,
        and his surprise at each of its word pieces"""
        import torch
        from transformers import DynamicCache
        ids, s, n = pre + span, len(pre), len(span)
        cache = DynamicCache()
        self.cap = {"h": [], "ple": {i: [] for i in range(L, N_LAYERS)}}
        surprise = []
        with torch.no_grad():
            for p0 in range(0, len(ids), CAP_CHUNK):
                chunk = ids[p0:p0 + CAP_CHUNK]
                lg = self._step(chunk, cache, dict(capture=True, s=s, n=n, pos0=p0), keep=0)[0]
                for r in range(len(chunk)):                        # position p0+r says what comes at p0+r+1
                    q = p0 + r + 1
                    if s <= q < s + n:
                        surprise.append(-torch.log_softmax(lg[r].float(), -1)[ids[q]].item())
                del lg
        mem = {"h": {L: torch.cat(self.cap["h"], 1)}, "ple": {i: torch.cat(t, 1) for i, t in self.cap["ple"].items()}}
        self.cap = None
        assert mem["h"][L].shape[1] == n and len(surprise) == n
        return mem, surprise

    def answer(self, ids, slots, max_new=MAX_NEW, chunk=CHUNK, split=None, at_split=None, raw=False):
        """greedy answer after reading ids (slots: [{"s", "n", "mem"}]); returns (text, ran out of room).
        split/at_split: read ids[:split] first, call at_split(cache) (the ogg squeeze of the old chat), then the rest.
        raw: keep the special markers (thinking mode's thought channel)"""
        import torch
        from transformers import DynamicCache
        cache = DynamicCache()
        P, said = len(ids), []
        cuts = [(0, P)] if split is None else [(0, split), (split, P)]
        with torch.no_grad():
            for a, b in cuts:
                for p0 in range(a, b, chunk):
                    lg = self._step(ids[p0:min(p0 + chunk, b)], cache, dict(slots=slots, pos0=p0))
                if at_split is not None and b == split:
                    at_split(cache)
            for k in range(max_new):
                t = int(lg[0, -1].argmax())
                if t in self.stop:
                    break
                said.append(t)
                lg = self._step([t], cache, dict(slots=slots, pos0=P + k))
        return self.tok.decode(said, skip_special_tokens=not raw).strip(), len(said) == max_new

    def next_scores(self, ids, slots, chunk=CHUNK):
        import torch
        from transformers import DynamicCache
        cache = DynamicCache()
        with torch.no_grad():
            for p0 in range(0, len(ids), chunk):
                lg = self._step(ids[p0:p0 + chunk], cache, dict(slots=slots, pos0=p0))
        return lg[0, -1].float()


# ---------------------------------------------------------------- his own summaries
def tidy(s):
    """his summary without a preface line ("Here's a shorter version:") or wrapping quotes"""
    s = s.strip()
    lines = s.split("\n")
    if len(lines) > 1 and lines[0].rstrip().endswith(":"):
        s = "\n".join(lines[1:]).strip()
    return re.sub(r'^"(.*)"$', r"\1", s, flags=re.S).strip()


def cmd_shorten(a):
    """every session of the bank shortened by him to half and a quarter of its word pieces (batched; eager attention,
    since HF's sdpa doesn't cover Gemma 4's audio part)"""
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation={"text_config": "sdpa", "vision_config": "eager", "audio_config": "eager"}).eval()
        print("text part on sdpa", flush=True)
    except Exception as e:
        print("sdpa per part refused, all eager:", str(e)[:120], flush=True)
        model = Gemma4ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager",
                                                               device_map="cuda").eval()
    stop = [tok.convert_tokens_to_ids("<turn|>"), tok.eos_token_id]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "shortened.jsonl"
    done = {(r["sid"], r["frac"]) for r in map(json.loads, open(path))} if path.exists() else set()
    ses = {s["sid"]: s for it in load_bank(a) for s in it["sessions"]}
    jobs = []
    for sid, s in ses.items():
        n = len(tok(s["text"], add_special_tokens=False)["input_ids"])
        for f in a.fracs.split(","):
            if (sid, f) not in done:
                k = math.ceil(FRAC[f] * n)
                row = tok("<bos>" + TURN + ASK.format(w=max(3, round(k * 0.75))) + s["text"].strip() + END,
                          add_special_tokens=False)["input_ids"]
                jobs.append((k, sid, f, n, row))
    jobs.sort(key=lambda j: (j[0], len(j[4])))
    print(f"{len(jobs)} summaries to write", flush=True)
    t0, f_out = time.time(), open(path, "a")
    i = 0
    while i < len(jobs):
        width = len(jobs[i][4]) + jobs[i][0] + 32
        b = max(1, min(32, a.budget // width))
        batch = jobs[i:i + b]
        i += b
        w = max(len(j[4]) for j in batch)
        x = torch.tensor([[tok.pad_token_id] * (w - len(j[4])) + j[4] for j in batch], device="cuda")
        m = torch.tensor([[0] * (w - len(j[4])) + [1] * len(j[4]) for j in batch], device="cuda")
        with torch.no_grad():
            y = model.generate(input_ids=x, attention_mask=m, max_new_tokens=max(j[0] for j in batch) + 32,
                               do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id)
        for j, row in zip(batch, y):
            said = tok.decode(row[w:], skip_special_tokens=True)
            ids = tok(tidy(said), add_special_tokens=False)["input_ids"][:j[0]]
            f_out.write(json.dumps(dict(sid=j[1], frac=j[2], n=j[3], k=j[0], said=said, ids=ids)) + "\n")
        f_out.flush()
        print(f"{i}/{len(jobs)} summaries, {time.time() - t0:.0f} s", flush=True)
    f_out.close()


# ---------------------------------------------------------------- the test
def f1(said, gold):
    """word overlap with the official answer (LoCoMo's F1): a pointer for reading, never the grade"""
    def norm(s):
        s = "".join(ch for ch in str(s).lower() if ch not in set(string.punctuation))
        return [w for w in s.split() if w not in ("a", "an", "the", "and")]
    p, g = norm(said), norm(gold)
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g or not same:
        return 0.0
    return round(2 * same / (len(p) + len(g)), 3)


def chat_a(rig, s):
    talk = random.Random(s["sid"]).sample(range(len(SMALL_TALK)), 2)
    return rig.ids(prefix(talk) + f"Session Date: {s['date']}\nSession Content:\n")


def chat_a_linked(rig, sessions, si):
    """chat A for session si with the earlier sessions of the same pile read first, as words: the related old
    memories are playing while the new one is recorded (Immy's "linked bricks", 2026-11-08). The words stand in for
    old memories recalled from floor 5, which answered like the words (step 6: 29 of 53 the same, letter for letter)"""
    s = sessions[si]
    talk = random.Random(s["sid"]).sample(range(len(SMALL_TALK)), 2)
    before = "".join(f"Session Date: {p['date']}\nSession Content:\n{p['text']}\n\n" for p in sessions[:si])
    return rig.ids(prefix(talk) + before + f"Session Date: {s['date']}\nSession Content:\n")


def cmd_run(a):
    import torch
    import rung0
    from sae import load_sae
    names = (a.only_ways.split(",") if a.only_ways else
             PRACTICE if a.ways == "practice" else [w.format(rule=a.rule) for w in TEST])
    for w in names:
        kind = split_way(w)[1].split("_")[0]
        if split_way(w)[0] and not (kind == "text" or kind.startswith("ogg") or kind in CARD_WAYS + MARK_WAYS):
            raise SystemExit(f"{w}: thinking is wired for the words, ogg, card and mark ways only")
        if kind.startswith("raw") and kind != f"raw{L}":
            raise SystemExit(f"{w} needs --floor {kind[3:]} (this run's floor is {L})")
        uses_finder = not (kind in WORDS_ONLY + ("cut", "sum") + CARD_WAYS + MARK_WAYS or
                           kind.startswith(("raw", "ogg")))
        if uses_finder and L != FINDER_FLOOR:
            raise SystemExit(f"{w} uses the finder, which is trained for floor {FINDER_FLOOR} only")
    items = load_bank(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers.jsonl"
    done = {(r["qid"], r["way"]) for r in map(json.loads, open(path))} if path.exists() else set()
    short = {}
    if any(w.split("_")[0] in ("sum", "sbr", "sbrq", "sleep") for w in names):
        for r in map(json.loads, open(a.short)):
            short[(r["sid"], r["frac"])] = r["ids"]
    rig, sae = Rig(a.model), load_sae(a.sae)
    codec, ogg_bits = None, {}
    squeezed = sorted({split_way(w)[1] for w in names if split_way(w)[1].startswith("ogg")} - {"oggraw"})
    if squeezed:
        import ogg
        codec = {k: {n: t.cuda() for n, t in c.items()} for k, c in torch.load(a.codec).items()}
        for b in squeezed:
            ogg_bits[b] = ogg.plan(codec, float(b[3:]))
            zero = sum(int((x == 0).sum()) for x in ogg_bits[b].values())
            total = sum(x.numel() for x in ogg_bits[b].values())
            print(f"{b}: {16 / float(b[3:]):.1f}x less room, {zero / total:.0%} of directions get no bits", flush=True)
    print("ways:", " ".join(names), flush=True)
    card_store, f_cards = {}, None
    if any(split_way(w)[1] in CARD_WAYS for w in names):
        cpath = out / "cards.jsonl"
        if cpath.exists():
            card_store = {r["key"]: r["cards"] for r in map(json.loads, open(cpath))}
        f_cards = open(cpath, "a")
    mark_store, f_marks = {}, None
    if any(split_way(w)[1] in MARK_WAYS for w in names):
        mpath = out / "marks.jsonl"
        if mpath.exists():
            mark_store = {r["qid"]: r for r in map(json.loads, open(mpath))}
        f_marks = open(mpath, "a")
    f_out, t0 = open(path, "a"), time.time()
    for it_no, it in enumerate(items):
        todo = [w for w in names if (it["qid"], w) not in done]
        if not todo:
            continue
        reps = {w: [] for w in todo}            # per way, per session: ("text", ids) or ("slot", k, mem)

        def summary_bricks(s, f):
            """his own f summary of session s, read once in its own chat A and kept as bricks"""
            sids = short[(s["sid"], f)]
            if not sids:
                return ("text", [])
            smem, _ = rig.capture(chat_a(rig, s), sids)
            with torch.no_grad():
                return ("slot", len(sids), {"h": {L: sae.roundtrip(smem["h"][L])}, "ple": smem["ple"]})

        cards = None
        if any(split_way(w)[1] in CARD_WAYS for w in todo):
            cards = write_cards(rig, it["sessions"], card_store, f_cards, a.chunk)
        marked = None
        if any(split_way(w)[1] in MARK_WAYS for w in todo):
            if it["qid"] not in mark_store:
                import marks
                tt, mm, notes = marks.mark_item(rig, it["sessions"])
                mark_store[it["qid"]] = dict(qid=it["qid"], times=tt, marks=mm, notes=notes)
                f_marks.write(json.dumps(mark_store[it["qid"]]) + "\n")
                f_marks.flush()
            marked = mark_store[it["qid"]]
        words_only = all(split_way(w)[1] in WORDS_ONLY + CARD_WAYS + MARK_WAYS or split_way(w)[1].startswith("ogg")
                         for w in todo)
        for si, s in enumerate(it["sessions"]):
            newest = si == len(it["sessions"]) - 1       # sessions are in date order
            span = rig.ids(s["text"])
            n = len(span)
            if words_only:                               # nothing here needs chat A's recording
                for w in todo:
                    b = split_way(w)[1]
                    rep = ([] if b == "none" else span + rig.ids(CARD_HEAD + cards[si]) if b == "cards" else
                           rig.ids(cards[si]) if b == "cardsonly" else rig.ids(marked[b][si]) if b in MARK_WAYS else
                           span)
                    reps[w].append(("text", rep))
                continue
            mem, surprise = rig.capture(chat_a(rig, s), span)
            ws = words(rig.tok, span)
            bricks = None
            if L == FINDER_FLOOR:
                with torch.no_grad():
                    bricks = {"h": {L: sae.roundtrip(mem["h"][L])}, "ple": mem["ple"]}
            for w in todo:
                kind, *rest = w.split("_")
                if split_way(w)[1] == "text" or split_way(w)[1].startswith("ogg"):   # ogg: room squeezed after reading
                    reps[w].append(("text", span))
                elif w == "none":
                    reps[w].append(("text", []))
                elif w == "bricks":
                    reps[w].append(("slot", n, bricks))
                elif kind == f"raw{L}":                    # the floor-L stream as read, no finder
                    reps[w].append(("slot", n, mem))
                elif kind == "sleep":                      # newest full, older ones his own notes, all as bricks
                    reps[w].append(("slot", n, bricks) if newest else summary_bricks(s, rest[0]))
                elif kind == "linked":                     # recorded while the earlier sessions were playing
                    if si == 0:
                        reps[w].append(("slot", n, bricks))
                    else:
                        lmem, _ = rig.capture(chat_a_linked(rig, it["sessions"], si), span)
                        with torch.no_grad():
                            reps[w].append(("slot", n, {"h": {L: sae.roundtrip(lmem["h"][L])}, "ple": lmem["ple"]}))
                        del lmem
                elif kind == "cutb":                       # his most surprising pieces kept as bricks, the rest dropped
                    keep = rung0.whole_words(rig.tok, span, surprise, FRAC[rest[0]])
                    reps[w].append(("slot", len(keep), {"h": {L: bricks["h"][L][:, keep]},
                                                        "ple": {i: x[:, keep] for i, x in bricks["ple"].items()}}))
                elif kind == "cut":
                    reps[w].append(("text", [span[j] for j in rung0.whole_words(rig.tok, span, surprise,
                                                                                FRAC[rest[0]])]))
                elif kind == "sum":
                    reps[w].append(("text", short[(s["sid"], rest[0])]))
                elif kind == "sbr":
                    reps[w].append(summary_bricks(s, rest[0]))
                elif kind == "sbrq":
                    sids = short[(s["sid"], rest[0])]
                    if not sids:
                        reps[w].append(("text", []))
                        continue
                    smem, ssur = rig.capture(chat_a(rig, s), sids)
                    groups = quiet_groups(words(rig.tok, sids), ssur, len(sids), 0.5, 4)
                    reps[w].append(("slot", len(groups), packed(sae, smem, groups, rest[1])))
                    del smem
                else:
                    if kind.startswith("pack"):
                        groups, rule = pack_groups(n, int(kind[4:])), rest[0]
                    else:
                        f, rule = rest
                        share = int(kind[5:]) if len(kind) > 5 else SHARE[f]
                        groups = quiet_groups(ws, surprise, n, FRAC[f], share)
                    reps[w].append(("slot", len(groups), packed(sae, mem, groups, rule)))
            del mem, bricks
        for w in todo:
            think, base = split_way(w)
            head = "<bos>" + (THINK_ON if think else "") + TURN + it["intro"]
            ids, slots, spots, n_all = rig.ids(head), [], 0, 0
            for k, (s, rep) in enumerate(zip(it["sessions"], reps[w])):
                ids += rig.ids(f"\n### Session {k + 1}:\nSession Date: {s['date']}\nSession Content:\n")
                if rep[0] == "text":
                    ids += rep[1]
                    spots += len(rep[1])
                else:
                    slots.append(dict(s=len(ids), n=rep[1], mem=rep[2]))
                    ids += [rig.pad] * rep[1]
                    spots += rep[1]
                ids += rig.ids("\n")
            history_end = len(ids)                           # the old chat ends here; the question comes next
            ids += rig.ids(it["outro"] + END)
            n_all = sum(len(rig.ids(s["text"])) for s in it["sessions"])
            split, squeeze = None, None
            if base.startswith("ogg"):                       # read the old chat, squeeze its room, then the question
                import ogg
                split = history_end
                if base != "oggraw":
                    squeeze = lambda cache, b=base: ogg.squeeze_cache(rig.model, cache, ogg.SINK, history_end, codec,
                                                                      ogg_bits[b])
            said, ran_out = rig.answer(ids, slots, a.think_max if think else a.max_new, a.chunk, split=split,
                                       at_split=squeeze, raw=think)
            extra = {}
            if think:                                        # thought kept apart from the answer, both saved
                full = said
                m = THOUGHT.match(full)
                thought, said = (m.group(1).strip(), m.group(2)) if m else ("", full)
                ran_out = ran_out or (m is not None and m.group(2) is None)   # still thinking when the room ran out
                said = re.sub(r"<\|?[a-z_]+\|?>", "", said or "").strip()
                extra = dict(thought=thought, thought_pieces=len(rig.ids(thought)), raw=full)
            f_out.write(json.dumps(dict(qid=it["qid"], qtype=it["qtype"], way=w, question=it["question"],
                                        gold=it["answer"], abstain=it["abstain"], said=said, ran_out=ran_out,
                                        spots=spots, n=n_all, prompt_len=len(ids), f1=f1(said, it["answer"]),
                                        **extra)) + "\n")
        f_out.flush()
        del reps
        torch.cuda.empty_cache()
        print(f"{it_no + 1}/{len(items)} questions, {time.time() - t0:.0f} s", flush=True)
    f_out.close()
    pointer(out)


def pointer(out):
    """per way: spots used and the word-overlap pointer (the real grade is read by hand into graded.jsonl)"""
    rows = [json.loads(l) for l in open(out / "answers.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["way"]].append(r)
    print(f"\n{'way':16}{'answers':>9}{'spots':>8}{'overlap':>9}{'ran out':>9}")
    for w, rs in by.items():
        print(f"{w:16}{len(rs):>9}{sum(r['spots'] for r in rs) / max(1, sum(r['n'] for r in rs)):>8.2f}"
              f"{sum(r['f1'] for r in rs) / len(rs):>9.3f}{sum(r['ran_out'] for r in rs):>9}")


# ---------------------------------------------------------------- check: the fast loop gives what the slow one gives
def cmd_check(a):
    """rung0.py's slow loop (whole prompt every word, one slot) is the reference the earlier results came from.
    The fast loop (chunked reading, cached answering, hiding at every step) must give the same answers, on the words
    and on bricks, and its splice must land (own stream put back = plain reading)."""
    import torch
    import rung0
    from sae import load_sae
    rung0.MAX_NEW = 24
    slow = rung0.Rig(a.model)
    fast, sae = Rig(base=slow), load_sae(a.sae)
    ms = json.load(open(a.memories))[:1]
    report = []
    for m in ms:
        span = fast.ids(m["text"])
        n = len(span)
        mem_a, _ = fast.capture(fast.ids(prefix(m["ctx_a"])), span)
        with torch.no_grad():
            bricks = {"h": {L: sae.roundtrip(mem_a["h"][L])}, "ple": mem_a["ple"]}
        pre = fast.ids(prefix(m["ctx_b"]))
        for q in m["questions"][:2]:
            tail = fast.ids("\n\n" + q["q"] + END)
            row_t, row_b = pre + span + tail, pre + [fast.pad] * n + tail
            slow_mem = {"h": {L: bricks["h"][L]}, "ple": bricks["ple"]}
            got = dict(
                text_slow=slow.answer(row_t, None), text_fast=fast.answer(row_t, [], 24, chunk=16)[0],
                bricks_slow=slow.answer(row_b, dict(s=len(pre), n=n, mem=slow_mem, L=L, block=True, ple=True)),
                bricks_fast=fast.answer(row_b, [dict(s=len(pre), n=n, mem=bricks)], 24, chunk=16)[0])
            got["same_text"] = got["text_slow"] == got["text_fast"]
            got["same_bricks"] = got["bricks_slow"] == got["bricks_fast"]
            report.append(got)
            print(json.dumps(got), flush=True)
        # the splice lands: the memory read in chat B itself, hidden below L, its own stream put back = plain reading
        mem_b, _ = fast.capture(pre, span)
        tail = fast.ids("\n\n" + m["questions"][0]["q"] + END)
        plain = fast.next_scores(pre + span + tail, [], chunk=16)
        back = fast.next_scores(pre + span + tail, [dict(s=len(pre), n=n, mem=mem_b)], chunk=16)
        hid = fast.next_scores(pre + span + tail, [dict(s=len(pre), n=n, mem={"h": {L: torch.zeros_like(
            mem_b["h"][L])}, "ple": mem_b["ple"]})], chunk=16)
        sp = dict(put_back_max_diff=(back - plain).abs().max().item(), hidden_max_diff=(hid - plain).abs().max().item())
        report.append(sp)
        print(json.dumps(sp), flush=True)
    json.dump(report, open(Path(a.out) / "check.json", "w"), indent=1)


def cmd_calib(a):
    """the ogg codec for his memory room, from public text rows (the finder's), never the test bank"""
    import numpy as np
    import torch
    import ogg
    rig = Rig(a.model)
    rows = torch.from_numpy(np.array(np.load(a.calib_rows, mmap_mode="r")[:a.calib_n]).astype(np.int64))
    t0 = time.time()
    codec = ogg.calibrate(rig.model, rows)
    torch.save({k: {n: t.cpu() for n, t in c.items()} for k, c in codec.items()}, a.codec)
    print(f"codec from {rows.numel()} word pieces in {time.time() - t0:.0f} s -> {a.codec}", flush=True)
    for avg in (8, 4, 2):
        bits = ogg.plan(codec, avg)
        for i in (0, 5, 11, 23):
            line = "  ".join(f"{kind}{g}: top {int(bits[(i, g, kind)][:8].sum())}/64 bits in 8 loudest, "
                             f"{int((bits[(i, g, kind)] == 0).sum())}/{bits[(i, g, kind)].numel()} silent"
                             for g in (0, 1) for kind in ("k", "v"))
            print(f"avg {avg} bits, floor {i}: {line}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "shorten", "run", "pointer", "calib"))
    ap.add_argument("--bank", choices=("lme", "locomo"))
    ap.add_argument("--data", help="the bank's json file")
    ap.add_argument("--model")
    ap.add_argument("--sae", help="the layer-8 fragment finder")
    ap.add_argument("--memories", help="check: richbank memories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ways", choices=("practice", "test"), default="practice")
    ap.add_argument("--rule", choices=RULES, help="how a shared spot is made (picked on the practice bank)")
    ap.add_argument("--short", help="run: shortened.jsonl from the shorten step")
    ap.add_argument("--fracs", default="h,q", help="shorten: which sizes (h half, q quarter, e eighth)")
    ap.add_argument("--budget", type=int, default=16384, help="shorten: word pieces per batch")
    ap.add_argument("--only-ways", help="run: these ways only, comma-separated (a small first test)")
    ap.add_argument("--qtypes", help="run: these question types only, comma-separated")
    ap.add_argument("--n", type=int, help="run: the first N questions only")
    ap.add_argument("--qids", help="a file of question ids, one per line (a fixed slice)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--think-max", type=int, default=THINK_MAX, help="run: room for thought + answer, think way")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--floor", type=int, help="run: the floor memories enter at (default: the finder's own floor)")
    ap.add_argument("--codec", help="calib: where to save the ogg codec; run: the codec for the oggN ways")
    ap.add_argument("--calib-rows", help="calib: public text rows (.npy, the finder's train.npy)")
    ap.add_argument("--calib-n", type=int, default=64, help="calib: how many rows of 512")
    a = ap.parse_args()
    global L, FINDER_FLOOR
    if a.sae:                                  # the finder file says which floor it was trained at
        import torch
        FINDER_FLOOR = torch.load(a.sae, map_location="cpu").get("layer", 8)
    L = a.floor if a.floor is not None else FINDER_FLOOR
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.cmd == "check":
        cmd_check(a)
    elif a.cmd == "shorten":
        cmd_shorten(a)
    elif a.cmd == "pointer":
        pointer(Path(a.out))
    elif a.cmd == "calib":
        cmd_calib(a)
    else:
        if a.ways == "test" and not a.rule:
            ap.error("--ways test needs --rule (picked on the practice bank)")
        cmd_run(a)


if __name__ == "__main__":
    main()
