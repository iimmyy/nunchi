"""Inputs for matching on all three jobs (2026-11-10).

The room's 4-bit brain was matched (tools/squeeze/match.py) on text inputs only (q4/act: llama-imatrix on the
calibration passages and Seoul's own translations). Measured per job (pieces-jobs/jobs.tsv), that file is better than
plain 4-bit on translation inputs (0.87x) but worse on hearing (1.18x) and voice (1.89x): matching fits what it sees.
This writes an act folder in match.py's layout (one f16 file of input rows per matrix input) with the three jobs
weighted equally:
  hearing      imi's recorder on 12 calibration-sermon clips (rec/hear)
  translation  the old text rows (q4/act) + imi's recorder on 24 calibration passages (rec/tr)
  voice        imi's recorder speaking 30 English sentences (rec/voice)
Each job's rows are scaled by sqrt(n_min / n_job), so every job adds the same weight to the matrix x^T x that
matching uses (a row scaled by s counts s^2). Rows of the jobs are interleaved so match.py's held-out rows (every 8th)
come from all three.

  python3 make_act3.py OLD_ACT REC OUT_ACT"""
import os, sys
from pathlib import Path
import numpy as np

old, rec, out = (Path(p) for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
SLOT = {"shortconv.in_proj": "in", "attn_q": "in", "shortconv.out_proj": "conv_out", "attn_output": "attn_out",
        "ffn_gate": "ffn_in", "ffn_down": "ffn_down"}


def rows(p, cols):
    return np.fromfile(p, dtype=np.float16).reshape(-1, cols)


def interleave(parts):
    """rows of several arrays mixed evenly: row k of part j goes to position (k + 0.5) / n_j"""
    keys = np.concatenate([(np.arange(len(a)) + 0.5) / len(a) for a in parts])
    order = np.argsort(keys, kind="stable")
    return np.concatenate(parts)[order]


for f in sorted(old.glob("blk.*.weight.f16")):
    il, kind = f.name.split(".")[1], f.name[len("blk.") + len(f.name.split(".")[1]) + 1:-len(".weight.f16")]
    slot = SLOT[kind]
    # the input width is the matrix's: ffn_down reads the 7680-wide activation, every other one the 2048-wide state
    cols = 7680 if kind == "ffn_down" else 2048
    text = rows(f, cols)
    hear, tr, voice = (rows(rec / j / f"L{il}.{slot}.f16", cols) for j in ("hear", "tr", "voice"))
    trans = interleave([text, tr])
    n = {"hear": len(hear), "translation": len(trans), "voice": len(voice)}
    nmin = min(n.values())
    parts = []
    for name, a in (("hear", hear), ("translation", trans), ("voice", voice)):
        parts.append((a.astype(np.float32) * np.sqrt(nmin / len(a))).astype(np.float16))
    x = interleave(parts)
    assert np.isfinite(x).all(), f.name
    x.tofile(out / f.name)
    print(f.name, cols, " ".join(f"{k} {v}" for k, v in n.items()), "->", len(x), "rows", flush=True)
