"""THROWAWAY scorer (2026-11-08): the clean sermon's hand-made Korean subtitles as the answer key.
Per window, the key = every subtitle line whose middle falls inside what that model heard (Gemma: the first 30 s of
the window; Omni: the whole window). CER = character edits / key characters, after dropping spaces and punctuation
(plain counting, no AI). Prints a per-window table and totals; writes a side-by-side read file for hand reading.
usage: score_clean.py KO.vtt OUT_READ.txt name=file.jsonl:field [name=file.jsonl:field ...]"""
import json
import re
import sys


def vtt(path):
    cues, t = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        m = re.match(r"(\d+):(\d+):(\d+\.\d+) --> (\d+):(\d+):(\d+\.\d+)", line)
        if m:
            h1, m1, s1, h2, m2, s2 = m.groups()
            t = (int(h1) * 3600 + int(m1) * 60 + float(s1), int(h2) * 3600 + int(m2) * 60 + float(s2))
            cues.append([t[0], t[1], ""])
        elif line and cues and not line.startswith(("WEBVTT", "Kind:", "Language:")):
            cues[-1][2] += (" " if cues[-1][2] else "") + line
    return cues


def norm(s):
    return re.sub(r"[^\w]|_", "", s or "")


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    cues = vtt(sys.argv[1])
    out = sys.argv[2]
    runs = []
    for arg in sys.argv[3:]:
        name, rest = arg.split("=", 1)
        path, field = rest.rsplit(":", 1)
        runs.append((name, field, {r["window"]: r for r in map(json.loads, open(path, encoding="utf-8"))}))
    wins = sorted(set.intersection(*[set(r) for _, _, r in runs]))
    tot = {name: [0, 0] for name, _, _ in runs}
    lines = []
    print("win  " + "  ".join(f"{n:>8}" for n, _, _ in runs))
    for w in wins:
        row, block = [], []
        for name, field, rs in runs:
            r = rs[w]
            end = r["start"] + r["heard_s"] if "heard_s" in r else r["end"]
            key = " ".join(c[2] for c in cues if r["start"] <= (c[0] + c[1]) / 2 < end)
            k, h = norm(key), norm(r.get(field))
            e = edits(h, k)
            tot[name][0] += e
            tot[name][1] += len(k)
            row.append(f"{100 * e / max(1, len(k)):7.0f}%")
            block.append(f"   {name} (CER {100 * e / max(1, len(k)):.0f}%, heard {r['start']:.0f}-{end:.0f}s): {r.get(field)}")
            if name == runs[0][0]:
                block.insert(0, f"   KEY: {key}")
        print(f"{w:3}  " + "  ".join(row))
        lines.append(f"\n#{w}\n" + "\n".join(block))
    print("ALL  " + "  ".join(f"{100 * e / n:7.1f}%" for e, n in tot.values()))
    open(out, "x", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
