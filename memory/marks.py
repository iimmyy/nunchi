"""Marks on the words at sleep (step 9 B, 2026-11-08; Immy: "anchor it in how a human decides on memories", then
"not just human like but better ... showcase my skills"): the human steps, done exactly.

1. when (hearing it): a person places "yesterday", "three weeks ago", "last month" on their own timeline at the
   moment they hear it (Friedman 1993). Here code does it from the chat's own date and writes the calendar date
   right after the words, in what the user said: "the Air Fryer I got yesterday [= 2023/05/20 (Sat)]". "≈" when the
   words are loose ("about three weeks ago", "a few weeks ago").
2. change (noticing it): the new fact that surprises him most with the older chats in mind (remind.py) is where the
   reminding happens (Wahlheim & Jacoby 2013). The older thing the user said that it changes is quoted right after
   it with its date, so the old fact sits inside the new memory in order.

Nothing is generated: dates are arithmetic, the change flag is his own surprise. The question is not touched.
"""
import calendar
import datetime as dt
import re

NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
       "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a couple": 2, "couple of": 2,
       "a few": 3, "few": 3, "several": 3}
LOOSE_NUM = {"a couple of", "a couple", "couple of", "a few", "few", "several"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
N = r"(\d+|a couple of|a couple|couple of|a few|few|several|an|a|one|two|three|four|five|six|seven|eight|nine|ten|" \
    r"eleven|twelve)"
APPROX = r"(?:(about|around|roughly|almost|nearly|over|just over|more than|less than|like) )?"
UNIT = r"(day|week|month|year)s?"


def chat_day(date):
    """LongMemEval's '2023/05/22 (Mon) 09:38' -> date"""
    return dt.datetime.strptime(date.split(" ")[0], "%Y/%m/%d").date()


def fmt(d):
    return f"{d:%Y/%m/%d} ({d:%a})"


def add(d, n, unit):
    if unit == "day":
        return d + dt.timedelta(days=n)
    if unit == "week":
        return d + dt.timedelta(weeks=n)
    m = d.month - 1 + n * (12 if unit == "year" else 1)
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def month(d):
    return f"{d:%Y/%m}"


def num(s):
    s = s.lower()
    return int(s) if s.isdigit() else NUM[s]


RULES = []


def rule(pattern):
    def deco(f):
        RULES.append((re.compile(pattern, re.I), f))
        return f
    return deco


@rule(r"\bthe day before yesterday\b")
def _(m, d):
    return "=", fmt(add(d, -2, "day"))


@rule(r"\bthe day after tomorrow\b")
def _(m, d):
    return "=", fmt(add(d, 2, "day"))


@rule(r"\b(yesterday|last night)\b")
def _(m, d):
    return "=", fmt(add(d, -1, "day"))


@rule(r"\b(today|this morning|this afternoon|this evening|tonight)\b")
def _(m, d):
    return "=", fmt(d)


@rule(r"\btomorrow\b")
def _(m, d):
    return "=", fmt(add(d, 1, "day"))


@rule(r"\b" + APPROX + N + r" " + UNIT + r" ago\b")
def _(m, d):
    n, unit = num(m.group(2)), m.group(3).lower()
    when = add(d, -n, unit)
    loose = m.group(1) or m.group(2).lower() in LOOSE_NUM
    return ("≈" if loose else "="), (fmt(when) if unit in ("day", "week") else month(when))


@rule(r"\bin " + APPROX + N + r" " + UNIT + r"\b")
def _(m, d):
    n, unit = num(m.group(2)), m.group(3).lower()
    when = add(d, n, unit)
    loose = m.group(1) or m.group(2).lower() in LOOSE_NUM
    return ("≈" if loose else "="), (fmt(when) if unit in ("day", "week") else month(when))


@rule(r"\b(last|next|this) (week|month|year|weekend)\b")
def _(m, d):
    which, unit = m.group(1).lower(), m.group(2).lower()
    step = {"last": -1, "next": 1, "this": 0}[which]
    if unit == "month":
        return "=", month(add(d.replace(day=1), step, "month"))
    if unit == "year":
        return "=", str(d.year + step)
    if unit == "week":
        mon = d - dt.timedelta(days=d.weekday()) + dt.timedelta(weeks=step)
        return "=", f"the week of {fmt(mon)}"
    sat = d + dt.timedelta(days=(5 - d.weekday()) % 7)            # this weekend: the coming (or current) Saturday
    if d.weekday() == 6:
        sat = d - dt.timedelta(days=1)
    return "=", f"the weekend of {fmt(sat + dt.timedelta(weeks=step))}"


@rule(r"\b(last|next|this past|this coming) (" + "|".join(DAYS) + r")\b")
def _(m, d):
    which, day = m.group(1).lower(), DAYS.index(m.group(2).lower())
    if which in ("last", "this past"):
        back = (d.weekday() - day) % 7 or 7
        return "=", fmt(d - dt.timedelta(days=back))
    ahead = (day - d.weekday()) % 7 or 7
    return "=", fmt(d + dt.timedelta(days=ahead))


@rule(r"\bfor " + APPROX + N + r" " + UNIT + r" now\b")
def _(m, d):
    n, unit = num(m.group(2)), m.group(3).lower()
    when = add(d, -n, unit)
    return "≈", "since " + (fmt(when) if unit in ("day", "week") else month(when))


def mark_times(text, date):
    """the same text with every relative time placed on the calendar, right after its words"""
    d = chat_day(date)
    hits = []
    for pat, f in RULES:
        for m in pat.finditer(text):
            if any(a < m.end() and m.start() < b for a, b, _ in hits):    # a longer rule already took these words
                continue
            sign, when = f(m, d)
            hits.append((m.start(), m.end(), f" [{sign} {when}]"))
    out, last = [], 0
    for a, b, note in sorted(hits):
        out += [text[last:b], note]
        last = b
    return "".join(out + [text[last:]]), len(hits)


def mark_session_times(turns, date):
    """a session's nl() text with times marked in what the user said (the assistant's words left as they are)"""
    out, n = "", 0
    for t in turns:
        say = t["content"].strip()
        if t["role"] == "user":
            say, k = mark_times(say, date)
            n += k
        out += f"\n\n{t['role']}: {say}"
    return out, n


# ---------------------------------------------------------------- the change: his surprise flags it, the old is quoted
STOP = set("""the a an and or but so to of in on at for with from by about as is are was were be been being am i me my
mine we us our you your he she it its they them their this that these those there here what which who whom when where
why how do does did done have has had having just also really very too not no yes can could would should will shall may
might must get got getting go going went like think thinking want wanted know actually way by_the lately recently now
then than some any all more most much many one ones thing things time one's i'm i've i'd i'll it's that's""".split())
SENT_END = re.compile(r"[.!?]+(?=\s|$)\s*|\n\s*")      # end marks before a space, or a line break


def sentences(s):
    """(start, end) of each sentence in s; together they are all of s, every character once"""
    cuts = sorted({0, len(s), *(m.end() for m in SENT_END.finditer(s))})
    return [(a, b) for a, b in zip(cuts, cuts[1:])] or [(0, len(s))]


def cue_words(s):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'&-]*", s.lower()) if w not in STOP and (len(w) > 2 or w.isdigit())}


def render(turns, date, add):
    """a session's nl() text with times marked in what the user said, and a reminding note after the sentences in
    add ((turn, sentence end) -> note). Also returns the user's sentences (as said, as marked) for the old store"""
    out, said = "", []
    for k, t in enumerate(turns):
        say = t["content"].strip()
        if t["role"] == "user":
            pieces = []
            for a, b in sentences(say):
                marked, _ = mark_times(say[a:b], date)
                tail = say[a:b][len(say[a:b].rstrip()):]
                pieces.append(marked.rstrip() + add.get((k, b), "") + tail)
                said.append((say[a:b], marked))
            say = "".join(pieces)
        out += f"\n\n{t['role']}: {say}"
    return out, said


def mark_item(rig, sessions, top=3, min_shared=2):
    """per session: its nl() text with times marked in what the user said (times), and the same with, in later chats,
    the old user sentence quoted after each of the user's most surprising sentences that shares its cue (marks).
    Returns (times, marks, notes)"""
    import shrinktest as st
    from remind import session_text
    old = []                                   # every earlier user sentence: (cue words, marked text, date)
    times, texts, notes = [], [], []
    for si, s in enumerate(sessions):
        turns = s["turns"]
        text, spans = session_text(turns)
        add = {}                               # (turn, sentence end within its content) -> note
        if si > 0 and old:
            enc = rig.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            span = enc["input_ids"]
            _, alone = rig.capture(st.chat_a(rig, s), span)
            _, linked = rig.capture(st.chat_a_linked(rig, sessions, si), span)
            best = {}                          # (turn, sentence) -> its biggest jump
            for j, (c0, _) in enumerate(enc["offset_mapping"]):
                k = next((i for i, t in enumerate(spans) if t["a"] <= c0 < t["b"]), None)
                if k is None or spans[k]["role"] != "user":
                    continue
                content = text[spans[k]["a"]:spans[k]["b"]]
                sid = next(i for i, (a, b) in enumerate(sentences(content)) if a <= c0 - spans[k]["a"] < b)
                best[(k, sid)] = max(best.get((k, sid), -1e9), linked[j] - alone[j])
            for (k, sid), jump in sorted(best.items(), key=lambda z: -z[1])[:top]:
                content = text[spans[k]["a"]:spans[k]["b"]]
                a, b = sentences(content)[sid]
                cues = cue_words(content[a:b])
                # the old sentence sharing the most cue words; a tie goes to the latest (the fact it most likely changes)
                i = max(range(len(old)), key=lambda i: (len(cues & old[i][0]), i))
                shared = cues & old[i][0]
                if len(shared) < min_shared:
                    continue
                add[(k, b)] = f' [earlier, on {old[i][2]}: "{old[i][1].strip()}"]'
                notes.append(dict(session=si + 1, jump=round(jump, 2), new=content[a:b].strip(),
                                  old=old[i][1].strip(), old_date=old[i][2], shared=sorted(shared)))
        plain, said = render(turns, s["date"], {})
        texts.append(render(turns, s["date"], add)[0])
        times.append(plain)
        old += [(cue_words(raw), m, s["date"].split(" ")[0]) for raw, m in said]   # cues from the words alone
    return times, texts, notes
