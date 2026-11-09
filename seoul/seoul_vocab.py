"""Seoul (LFM3-Omni-2B) word table: which pieces are Korean, English, digits and punctuation?

  python3 seoul_vocab.py count HF_DIR            # how big a Korean + English table would be
  python3 seoul_vocab.py keep HF_DIR OUT.json    # the keep list (old ids, sorted), closed under BPE merges

A piece is kept when its text (bytes decoded) uses only Hangul, Latin, Greek, Hebrew (sermons quote them) or common characters (digits, punctuation,
symbols, spaces). Every special token and all 256 single bytes are kept, so any text can still be written.
Then the set is closed under the merge rules: a kept piece also keeps the two pieces it is merged from, or the
tokenizer could never reach it.
"""
import json, sys, unicodedata
from collections import Counter


def bytes_to_unicode():  # GPT-2 byte-level alphabet
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


U2B = bytes_to_unicode()


def piece_bytes(p):
    return bytes(U2B[c] for c in p)


def script(ch):
    if ch.isspace() or not ch.isalpha():
        return "common"
    name = unicodedata.name(ch, "")
    for s in ("HANGUL", "LATIN", "CJK", "CYRILLIC", "ARABIC", "HIRAGANA", "KATAKANA", "DEVANAGARI", "THAI", "GREEK",
              "HEBREW"):
        if name.startswith(s) or f" {s} " in f" {name} ":
            return s
    return "other"


def classify(p):
    """'byte' for single bytes, 'partial' for pieces that are not whole UTF-8 text, else the set of scripts used"""
    b = piece_bytes(p)
    if len(b) == 1:
        return "byte"
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        return "partial"
    s = {script(c) for c in text} - {"common"}
    return s or {"common"}


def load(hf):
    t = json.load(open(f"{hf}/tokenizer.json", encoding="utf-8"))
    vocab = t["model"]["vocab"]
    merges = [m.split(" ") if isinstance(m, str) else m for m in t["model"]["merges"]]
    added = {a["id"]: a["content"] for a in t["added_tokens"]}
    return vocab, merges, added


def keep_set(hf):
    vocab, merges, added = load(hf)
    ok = {"HANGUL", "LATIN", "GREEK", "HEBREW", "common"}   # sermons quote Greek and Hebrew words
    keep, why = set(added), Counter()
    for p, i in vocab.items():
        c = classify(p)
        if c == "byte":
            keep.add(i); why["single byte"] += 1
        elif c == "partial":   # cuts across a character; Korean text uses these a lot (e.g. " 이" + half of 삭)
            keep.add(i); why["partial (cuts across a character)"] += 1
        elif c <= ok:
            keep.add(i); why["+".join(sorted(c))] += 1
        else:
            why["dropped: " + "+".join(sorted(c))] += 1
    # merge closure: walk the merges until nothing new is added
    made = {}
    for a, b in merges:
        made[a + b] = (a, b)
    inv = {i: p for p, i in vocab.items()}
    todo = [inv[i] for i in keep if i in inv]
    while todo:
        p = todo.pop()
        if p in made:
            for q in made[p]:
                j = vocab[q]
                if j not in keep:
                    keep.add(j); why["added by merge closure"] += 1
                    todo.append(q)
    return sorted(keep), why, len(vocab) + len(added)


def main():
    cmd, hf = sys.argv[1:3]
    keep, why, total = keep_set(hf)
    cfg = json.load(open(f"{hf}/config.json"))
    rows = (cfg.get("text_config") or cfg).get("vocab_size")
    for k, v in why.most_common():
        print(f"{v:8d}  {k}")
    print(f"kept {len(keep)} of {total} pieces; the table has {rows} rows -> {len(keep) / rows:.1%} of the rows")
    if cmd == "keep":
        json.dump({"keep": keep, "rows_before": rows}, open(sys.argv[3], "w"))
        print("wrote", sys.argv[3])


if __name__ == "__main__":
    main()
