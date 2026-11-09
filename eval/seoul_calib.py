"""Calibration text for Seoul's 4-bit pipeline, in Seoul's own chat format and its real jobs.

  python3 seoul_calib.py SUBS_DIR OUT_DIR [--n 240] [--port 8092] [--seed 0]

Korean passages (~25 s of subtitles each) are drawn from the TRAINING sermons' subtitles (never the held-out test
sermon). A running llama-server with 16-bit Seoul translates each one to English (greedy). Writes:
  pairs.jsonl  every passage with Seoul's English (kept as evidence; also a first look at the translations)
  calib.txt    ChatML turns: the ASR job (system ASR prompt, Korean answer) and the translation job (Korean in,
               English out), alternating, for llama-imatrix --parse-special
The audio itself cannot go through llama-imatrix, so the hearing job is represented by its Korean answer.
"""
import json, os, random, re, sys, urllib.request

ASR = "Perform ASR in Korean."
TEXT_TRANSLATE = "Please translate the following Korean into English text."
TEST_SERMON = os.environ.get("NUNCHI_TEST_ID", "")  # file-name prefix of the held-out sermon, never used for calibration


def vtt(path):
    cues = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        m = re.match(r"(\d+):(\d+):(\d+\.\d+) --> (\d+):(\d+):(\d+\.\d+)", line)
        if m:
            h1, m1, s1, h2, m2, s2 = m.groups()
            cues.append([int(h1) * 3600 + int(m1) * 60 + float(s1), int(h2) * 3600 + int(m2) * 60 + float(s2), ""])
        elif line and cues and not line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            cues[-1][2] += (" " if cues[-1][2] else "") + re.sub(r"<[^>]+>", "", line)
    return [c for c in cues if c[2].strip()]


def passages(subs_dir, max_s=25.0):
    out = []
    for f in sorted(os.listdir(subs_dir)):
        if not f.endswith(".vtt") or (TEST_SERMON and f.startswith(TEST_SERMON)):
            continue
        cur = []
        for c in vtt(os.path.join(subs_dir, f)) + [None]:
            if cur and (c is None or c[1] - cur[0][0] > max_s or c[0] - cur[-1][1] > 3.0):
                out.append((f.split(".")[0], " ".join(x[2] for x in cur)))
                cur = []
            if c is not None:
                cur.append(c)
    return out


def translate(port, ko):
    body = {"messages": [{"role": "system", "content": TEXT_TRANSLATE}, {"role": "user", "content": ko}],
            "temperature": 0, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.loads(r.read())
    return res["choices"][0]["message"]["content"].strip(), res["choices"][0].get("finish_reason")


def turn(system, user, answer):
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    if user is not None:
        s += f"<|im_start|>user\n{user}<|im_end|>\n"
    return s + f"<|im_start|>assistant\n<think></think>{answer}<|im_end|>\n"


def main():
    subs, out_dir = sys.argv[1:3]
    rest = sys.argv[3:]
    get = lambda k, d: type(d)(rest[rest.index(k) + 1]) if k in rest else d
    n, port, seed = get("--n", 240), get("--port", 8092), get("--seed", 0)
    allp = passages(subs)
    pick = random.Random(seed).sample(allp, min(n, len(allp)))
    os.makedirs(out_dir, exist_ok=True)
    pairs_path = os.path.join(out_dir, "pairs.jsonl")
    done = {}
    if os.path.exists(pairs_path):
        for l in open(pairs_path, encoding="utf-8"):
            r = json.loads(l)
            done[r["ko"]] = r
    with open(pairs_path, "a", encoding="utf-8") as fo:
        for i, (sid, ko) in enumerate(pick):
            if ko in done:
                continue
            en, why = translate(port, ko)
            r = {"sermon": sid, "ko": ko, "en": en, "finish": why}
            done[ko] = r
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            print(i + 1, len(pick), en[:70], flush=True)
    with open(os.path.join(out_dir, "calib.txt"), "x", encoding="utf-8") as fc:
        for sid, ko in pick:
            r = done[ko]
            fc.write(turn(ASR, None, ko))
            if r["finish"] != "length":           # a translation that ran out is not a real answer
                fc.write(turn(TEXT_TRANSLATE, ko, r["en"]))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
