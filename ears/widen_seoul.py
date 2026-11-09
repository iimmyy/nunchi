#!/usr/bin/env python3
"""Wide ears for Seoul (our translator, built on LiquidAI LFM3-Omni-2B): hearing from 8 up to ~21 kHz.

The same idea as widen.py (Gemma's wide ears), ported to Seoul's ears. Seoul hears 16 kHz sound as 128 Slaney mel
bands (0-8 kHz, 10 ms steps), squeezed by a small conv stack (Lfm3OmniDownsampler: three stride-2 convs, 8 frames
per token) and read by a 17-layer encoder. This adds a side path, like a LoRA on the input:
  - the old path is untouched: it hears the sound resampled to 16 kHz, exactly as before
  - a new front end takes the 48 kHz sound in the same 10 ms steps and makes 40 more bands from 8 kHz up, spaced the
    way Seoul's 128 are (Slaney mel, the same step; the first starts where the top old band ends), scaled and
    floored the way Seoul's front end does it (log10, floor -12, (x + 4) / 4)
  - a copy of Seoul's conv stack reads those bands (its convs start as copies of the old ones), and an output layer
    that starts at zero adds the result to the old path's tokens before the encoder, so on day one Seoul hears
    exactly what it heard
Only the side path trains; the old ears, the encoder and the brain stay frozen.

What it learns from: Korean speech with hand-made subtitles (lesson = "Perform ASR in Korean." -> the subtitle
text). Half the lessons go through a simulated big room (an exponential reverb tail, RT60 0.8-1.4 s, direct sound
0-12 dB below the room, plus a little noise), so the worst case trains too.
--cut8: the control run. Every training clip is cut at 8 kHz first, so the side path sees no real high end; it shows
how much of the gain is the training itself rather than the new hearing.

  clips  SUBS_DIR OUT.jsonl                        subtitle-aligned clips (<= 25 s) from SUBS_DIR/*.48k.wav + *.vtt
  irs    CLAPS.wav OUT.npy T1,T2,...               real room impulse responses from hand claps at those seconds
         (train --irs OUT.npy: half of the room lessons use them)
  check  MODEL WAV48                               day one: tokens and answer with the side path on = off (exactly)
  train  MODEL CLIPS.jsonl OUT_DIR [--steps 1200] [--lr 3e-4] [--accum 4] [--room 0.5] [--cut8] [--seed 0]
         [--workers 2]   (the lessons' sound is made on CPU workers ahead of the GPU)
  asr    MODEL WAV48 OUT.jsonl [--side OUT_DIR/side.pt] [--n 0] [--ir IRS.npy:INDEX]   pause-cut windows, greedy
         ASR, side on (or off); --ir plays the recording through that room first
"""
import copy
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import fftconvolve, resample_poly, butter, sosfiltfilt

SR = 48000
N_NEW = 40
ASR = "Perform ASR in Korean."
MAX_S = 25.0


def hz_to_mel(f):                                         # Slaney (the scale Seoul's front end uses)
    f = np.asarray(f, dtype=np.float64)
    ml, ls = 15.0, math.log(6.4) / 27.0
    return np.where(f >= 1000.0, ml + np.log(np.maximum(f, 1e-9) / 1000.0) / ls, 3.0 * f / 200.0)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    ml, ls = 15.0, math.log(6.4) / 27.0
    return np.where(m >= ml, 1000.0 * np.exp(ls * (m - ml)), 200.0 * m / 3.0)


def high_filters(n_fft=1200, sr=SR, n_new=N_NEW):
    """n_new triangles continuing Seoul's 128 (same mel step, first lower edge at 8 kHz), Slaney area-normalized"""
    step = (hz_to_mel(8000.0) - hz_to_mel(0.0)) / 129.0
    edges = mel_to_hz(hz_to_mel(8000.0) + step * np.arange(n_new + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_new, len(freqs)))
    for i in range(n_new):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up, down = (freqs - lo) / (c - lo), (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0, np.minimum(up, down)) * 2.0 / (hi - lo)
    return torch.tensor(fb, dtype=torch.float32), edges


class HighBands(nn.Module):
    """48 kHz sound -> (n_new, T) features in Seoul's style; T matches the old front end's frame count.
    The window is 25 ms like Seoul's (1200 samples at 48 kHz vs 400 at 16 kHz), so power per band comes out 9x
    larger for noise-like sound (3x per bin, 3x as many bins); dividing by 9 puts the new bands on the old scale."""

    def __init__(self, n_new=N_NEW):
        super().__init__()
        fb, self.edges = high_filters(n_new=n_new)
        self.register_buffer("fb", fb)
        self.register_buffer("win", torch.hann_window(1200))

    def forward(self, y48, n_frames):
        need = 1200 + 480 * (n_frames - 1)
        y = F.pad(y48, (0, max(0, need - y48.shape[-1])))[..., :need]
        spec = torch.stft(y, n_fft=1200, hop_length=480, win_length=1200, window=self.win, center=False,
                          return_complex=True)
        power = spec.abs().square() / 9.0
        mel = (self.fb @ power).clamp_min(1e-30).log10().clamp(min=-12.0)
        return (mel + 4.0) / 4.0


class Side(nn.Module):
    """a copy of Seoul's conv stack for the new bands; its output layer starts at zero"""

    def __init__(self, ds, n_new=N_NEW):
        super().__init__()
        for name in ("conv_in", "conv_mid", "lin_mid", "conv_end", "lin_end"):
            setattr(self, name, copy.deepcopy(getattr(ds, name)).float())
        rows = n_new
        for _ in range(3):
            rows = (rows + 1) // 2
        ch = ds.conv_in.out_channels
        self.lin_out = nn.Conv1d(rows * ch, ds.lin_out.out_channels, kernel_size=1)
        nn.init.zeros_(self.lin_out.weight)
        nn.init.zeros_(self.lin_out.bias)
        self.act = nn.SiLU()

    def forward(self, hi, lengths=None):
        def mask(h, n):
            if n is None:
                return h
            ok = torch.arange(h.shape[-1], device=h.device)[None] < n[:, None]
            return h * ok[:, None, None].to(h.dtype)
        h = mask(hi.unsqueeze(1), lengths)
        h = self.act(self.conv_in(h))
        n = None if lengths is None else (lengths + 1) // 2
        h = mask(h, n)
        h = self.act(self.lin_mid(self.conv_mid(h)))
        n = None if n is None else (n + 1) // 2
        h = mask(h, n)
        h = self.act(self.lin_end(self.conv_end(h)))
        return self.lin_out(h.flatten(1, 2))


def attach(model):
    """puts the side path on Seoul's ears; set model._high (1, n_new, T) before each forward (None = old ears)"""
    ds = model.model.audio_encoder.downsample
    side = Side(ds).to(ds.conv_in.weight.device)
    old = ds.forward

    def forward(input_features, feature_lengths=None):
        h = old(input_features, feature_lengths)
        hi = getattr(model, "_high", None)
        if hi is not None:
            add = side(hi.to(h.device, torch.float32), feature_lengths)
            add = add[..., : h.shape[-1]]
            if getattr(model, "_probe", None) is not None:   # size of the old ears' rows and of the side's push
                model._probe.append((h.float().norm(dim=1).mean().item(), add.float().norm(dim=1).mean().item(),
                                     float(hi.float().mean()), float(hi.float().std())))
            if getattr(model, "_push", None) is not None:    # training: the push as a share of the old rows
                model._push.append((add.float().norm(dim=1) / (h.float().norm(dim=1) + 1e-6)).mean())
            h = h + add.to(h.dtype)
        return h
    ds.forward = forward
    model._side, model._high = side, None
    return side


def load(model_dir, side_pt=None):
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    for p in model.parameters():
        p.requires_grad_(False)
    side = attach(model)
    if side_pt:
        side.load_state_dict(torch.load(side_pt, map_location="cuda")["side"])
    return proc, model, side


def to16(y48):
    return resample_poly(y48, 1, 3).astype(np.float32)


def encode(proc, model, hfe, y48, answer=None, use_side=True, system=ASR):
    """the chat (system, then the sound), old features from the 16 kHz version, new bands from the 48 kHz one;
    with an answer: labels on the answer's tokens only"""
    y16 = to16(y48)
    conv = [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    full = prompt + (answer + "<|im_end|>" if answer is not None else "")
    enc = proc(text=[full], audio=[y16], sampling_rate=16000, return_tensors="pt").to(model.device)
    model._high = None
    if use_side:
        T = enc["input_features"].shape[-1]
        hi = hfe(torch.from_numpy(y48).to(model.device), T)[None]
        m = enc["feature_attention_mask"]
        model._high = hi * m[:, None, :].to(hi.dtype)
    labels = None
    if answer is not None:
        ids = enc["input_ids"][0]
        tail = proc.tokenizer(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if ids[-len(tail):].tolist() == tail:                 # the answer tokenized on its own the same way
            n_prompt = len(ids) - len(tail)
        else:                                                 # rare: run the prompt alone (costs a second mel pass)
            n_prompt = proc(text=[prompt], audio=[y16], sampling_rate=16000, return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100
    return enc, labels


@torch.no_grad()
def transcribe(proc, model, hfe, y48, use_side, max_new=512):
    enc, _ = encode(proc, model, hfe, y48, use_side=use_side)
    out = model.generate(**enc, max_new_tokens=max_new, generation_mode="text", do_sample=False)
    n_in = enc["input_ids"].shape[1]
    end = int(out.sequence_lengths[0]) if hasattr(out, "sequence_lengths") else out.sequences.shape[1]
    ids = out.sequences[0, n_in:end]
    return proc.decode(ids, skip_special_tokens=True).strip(), len(ids) >= max_new


# ---------- data ----------

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


def clips(subs_dir, out):
    """consecutive subtitle lines grouped into clips of at most 25 s (no gap over 3 s inside one)"""
    rows = []
    for f in sorted(os.listdir(subs_dir)):
        if not f.endswith(".vtt"):
            continue
        sid = f.split(".")[0]
        wav = next((p for p in (os.path.join(subs_dir, sid + e) for e in (".48k.wav", ".flac")) if os.path.exists(p)), None)
        if wav is None:
            continue
        cues, cur = vtt(os.path.join(subs_dir, f)), []
        for c in cues + [None]:
            if cur and (c is None or c[1] - cur[0][0] > MAX_S or c[0] - cur[-1][1] > 3.0):
                rows.append({"sermon": sid, "wav": wav, "t0": max(0.0, cur[0][0] - 0.2), "t1": cur[-1][1] + 0.2,
                             "text": " ".join(x[2] for x in cur)})
                cur = []
            if c is not None:
                cur.append(c)
    with open(out, "x", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"clips": len(rows), "sermons": len({r['sermon'] for r in rows}),
                      "hours": round(sum(r["t1"] - r["t0"] for r in rows) / 3600, 2)}))


def read_clip(r):
    a, b = int(r["t0"] * SR), int(r["t1"] * SR)
    y, sr = sf.read(r["wav"], start=a, stop=b, dtype="float32")
    assert sr == SR
    y = y.mean(1) if y.ndim > 1 else y
    return y[: len(y) // 3 * 3]


def irs(wav48, out, times):
    """real room impulse responses from hand claps recorded in the room (Immy's idea: the clap as a convolver):
    each clap from 5 ms before its peak to 1.5 s after, energy-normalized. A phone recording also carries the
    phone's own cut (no sound above ~12 kHz), so convolving with these gives room + phone in one step."""
    y, sr = sf.read(wav48, dtype="float32")
    assert sr == SR
    y = y.mean(1) if y.ndim > 1 else y
    out_irs = []
    for t in (float(s) for s in times.split(",")):
        a = int(t * SR)
        seg = y[a - int(0.1 * SR): a + int(0.1 * SR)]
        p = a - int(0.1 * SR) + int(np.argmax(np.abs(seg)))
        ir = y[p - int(0.005 * SR): p + int(1.5 * SR)].astype(np.float64)
        out_irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    np.save(out, np.stack(out_irs))
    print(json.dumps({"irs": len(out_irs), "samples": len(out_irs[0])}))


IRS = None


TAIL_BANDS = [0, 250, 500, 1000, 2000, 4000, 6000, 8000, 11000, 15000, 24000]
FLAT_ROOMS = False   # --flat-rooms: round 1's flat white tail and white noise (kept only to reproduce round 1)


def damped_tail(rng, rt):
    """a room tail whose highs die faster than its lows, like real rooms (and like the church claps: under 0.3% of
    the tail's energy above 8 kHz). Per band: RT60 = rt below 1 kHz, falling as (1 kHz / f) ** a above it
    (a = 0.5-0.9), and the walls take -3 to -6 dB per octave above 4 kHz from the start."""
    g = np.random.default_rng(rng.randrange(1 << 30))
    n = int(SR * rt * 1.2)
    t = np.arange(n) / SR
    a, tilt = rng.uniform(0.5, 0.9), rng.uniform(3, 6)
    tail = np.zeros(n)
    for lo, hi in zip(TAIL_BANDS[:-1], TAIL_BANDS[1:]):
        X = np.fft.rfft(g.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / SR)
        X[(f < lo) | (f >= hi)] = 0
        fc = max(np.sqrt(max(lo, 20) * hi), 20)
        rt_b = rt * (min(1.0, 1000 / fc) ** a)
        gain = 10 ** (-tilt * max(0.0, np.log2(fc / 4000)) / 20)
        tail += np.fft.irfft(X, n) * np.exp(-6.9 * t / rt_b) * gain
    return tail


def room_noise(rng, n):
    """room tone: pink (-3 dB per octave) and rolled off above 8-12 kHz, not flat white"""
    g = np.random.default_rng(rng.randrange(1 << 30))
    X = np.fft.rfft(g.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    fc = rng.uniform(8000, 12000)
    X *= 1 / np.sqrt(np.maximum(f, 20)) / np.sqrt(1 + (f / fc) ** 8)
    x = np.fft.irfft(X, n)
    return x / (np.std(x) + 1e-12)


def room(y, rng):
    """a big room. With real claps loaded (--irs), half the rooms are those claps; the rest: a damped tail
    RT60 0.8-1.4 s (at the lows), direct sound 0-12 dB below the room. A little room tone either way."""
    noise = (lambda n: np.random.default_rng(rng.randrange(1 << 30)).standard_normal(n)) if FLAT_ROOMS else \
        (lambda n: room_noise(rng, n))
    if IRS is not None and rng.random() < 0.5:
        wet = fftconvolve(y, IRS[rng.randrange(len(IRS))])[: len(y)]
        wet += noise(len(wet)) * np.std(wet) * 10 ** (-rng.uniform(20, 40) / 20)
        return (wet / (np.abs(wet).max() + 1e-9) * 0.5).astype(np.float32)
    rt = rng.uniform(0.8, 1.4)
    if FLAT_ROOMS:
        t = np.arange(int(SR * rt * 1.2)) / SR
        tail = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(t)) * np.exp(-6.9 * t / rt)
    else:
        tail = damped_tail(rng, rt)
    tail[: int(0.003 * SR)] = 0
    tail /= np.sqrt(np.sum(tail ** 2))
    direct = np.zeros_like(tail)
    direct[0] = 10 ** (-rng.uniform(0, 12) / 20)
    wet = fftconvolve(y, direct + tail)[: len(y)]
    wet += noise(len(wet)) * np.std(wet) * 10 ** (-rng.uniform(15, 35) / 20)
    return (wet / (np.abs(wet).max() + 1e-9) * 0.5).astype(np.float32)


CUT8 = butter(12, 7800, btype="low", fs=SR, output="sos")


def loud(y, target_db=-20.0):
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-9
    g = min(10 ** (target_db / 20) / rms, 0.99 / (float(np.abs(y).max()) + 1e-9))
    return (y * g).astype(np.float32)


# ---------- commands ----------

@torch.no_grad()
def check(model_dir, wav48):
    proc, model, side = load(model_dir)
    hfe = HighBands().cuda()
    y, sr = sf.read(wav48, dtype="float32")
    y = (y.mean(1) if y.ndim > 1 else y)[: 20 * SR]
    y = loud(y[: len(y) // 3 * 3])
    enc, _ = encode(proc, model, hfe, y, use_side=False)
    off = model.model.get_audio_features(enc["input_features"], enc["feature_attention_mask"])
    enc, _ = encode(proc, model, hfe, y, use_side=True)
    on = model.model.get_audio_features(enc["input_features"], enc["feature_attention_mask"])
    hi = model._high[0]
    a, ra = transcribe(proc, model, hfe, y, False)
    b, rb = transcribe(proc, model, hfe, y, True)
    # where the new bands sit next to the old top bands, on this recording
    old_top = enc["input_features"][0, -4:].float().mean().item()
    print(json.dumps({"tokens_max_diff": (on.float() - off.float()).abs().max().item(), "same_answer": a == b,
                      "answer": a[:200], "old_top4_mean": round(old_top, 3),
                      "new_low4_mean": round(hi[:4].mean().item(), 3), "new_high4_mean": round(hi[-4:].mean().item(), 3),
                      "edges_khz": [round(e / 1000, 2) for e in list(HighBands().edges[[0, 1, -1]])]}, ensure_ascii=False))


def _worker_init(irs_path, flat_rooms=False):
    global IRS, FLAT_ROOMS
    IRS = np.load(irs_path) if irs_path else None
    FLAT_ROOMS = flat_rooms
    torch.set_num_threads(1)


def _make(job):
    """one lesson's sound, made on a CPU worker so the GPU never waits: read, (cut at 8 kHz), (room), level"""
    r, seed, cut8, p_room = job
    rng = random.Random(seed)
    y = read_clip(r)
    if cut8:                                              # brick wall at 7.8 kHz (one FFT; nothing above survives)
        Y = np.fft.rfft(y)
        Y[int(7800 * len(y) / SR):] = 0
        y = np.fft.irfft(Y, len(y)).astype(np.float32)
    if rng.random() < p_room:
        y = room(y, rng)
    return r["text"], loud(y)


def train(model_dir, clips_path, out_dir, *rest):
    flags = {"--cut8", "--flat-rooms"}
    vals = [a for a in rest if a not in flags]
    kw = {vals[i][2:]: vals[i + 1] for i in range(0, len(vals), 2)}
    cut8, flat_rooms = "--cut8" in rest, "--flat-rooms" in rest
    steps, lr, accum = int(kw.get("steps", 1200)), float(kw.get("lr", 3e-4)), int(kw.get("accum", 4))
    p_room, seed = float(kw.get("room", 0.5)), int(kw.get("seed", 0))
    # the side path may add at most push_cap of the old ears' row size before it pays (round 1 grew to 0.45-0.62)
    push_cap, push_w = float(kw.get("push-cap", 0.15)), float(kw.get("push-w", 10.0))
    global IRS
    if kw.get("irs"):
        IRS = np.load(kw["irs"])
    os.makedirs(out_dir, exist_ok=False)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    proc, model, side = load(model_dir)
    hfe = HighBands().cuda()
    rows = [json.loads(l) for l in open(clips_path, encoding="utf-8")]
    side.train()
    opt = torch.optim.AdamW(side.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    log = open(os.path.join(out_dir, "train.jsonl"), "w")
    cfg = {"steps": steps, "lr": lr, "accum": accum, "room": p_room, "cut8": cut8, "flat_rooms": flat_rooms, "seed": seed, "clips": len(rows),
           "irs": kw.get("irs"), "push_cap": push_cap, "push_w": push_w}
    print(json.dumps(cfg), flush=True)
    import multiprocessing as mp
    jobs = ((rng.choice(rows), rng.randrange(1 << 30), cut8, p_room) for _ in range(steps * accum))
    pool = mp.get_context("spawn").Pool(int(kw.get("workers", 2)), initializer=_worker_init,
                                     initargs=(kw.get("irs"), flat_rooms))
    # Bounded read-ahead. pool.imap queues every job at once and keeps every finished clip in memory, so when the
    # workers outrun the GPU the backlog grows until the box runs out of RAM (cut8b was OOM-killed at step ~700).
    from collections import deque
    ahead = deque(pool.apply_async(_make, (next(jobs),)) for _ in range(4 * accum))

    def next_clip():
        r = ahead.popleft()
        nxt = next(jobs, None)
        if nxt is not None:
            ahead.append(pool.apply_async(_make, (nxt,)))
        return r.get()

    t0, run, pushes = time.time(), [], []
    for step in range(steps):
        for _ in range(accum):
            text, y = next_clip()
            enc, labels = encode(proc, model, hfe, y, answer=text)
            model._push = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits
            loss = F.cross_entropy(logits[0, :-1].float(), labels[0, 1:], ignore_index=-100)
            push = torch.stack(model._push).mean()
            model._push = None
            ((loss + push_w * F.relu(push - push_cap) ** 2) / accum).backward()
            run.append(loss.item())
            pushes.append(push.item())
        torch.nn.utils.clip_grad_norm_(side.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 20 == 0:
            rec = {"step": step + 1, "loss": round(float(np.mean(run)), 4), "s": round(time.time() - t0),
                   "out_norm": round(side.lin_out.weight.norm().item(), 4), "push": round(float(np.mean(pushes)), 4)}
            run, pushes = [], []
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if (step + 1) % 400 == 0 or step + 1 == steps:
            torch.save({"side": side.state_dict(), "cfg": cfg, "step": step + 1},
                       os.path.join(out_dir, "side.pt" if step + 1 == steps else f"side-{step + 1}.pt"))
    pool.terminate()


def cut_points(x, sr, every=30.0, search=5.0, quiet=0.3):
    """the same pause-cut windows as every translator test (omni_spike.cut_points)"""
    hop = int(0.02 * sr)
    rms = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(hop) / hop, mode="same")[::hop])
    k = max(1, int(quiet / 0.02))
    smooth = np.convolve(rms, np.ones(k) / k, mode="same")
    edges, t = [0], every
    total = len(x) / sr
    while t < total - 5:
        a, b = int((t - search) / 0.02), int(min(t + search, total) / 0.02)
        j = a + int(np.argmin(smooth[a:b]))
        edges.append(j * hop)
        t = j * 0.02 + every
    edges.append(len(x))
    return edges


def asr(model_dir, wav48, out, *rest):
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    proc, model, side = load(model_dir, kw.get("side"))
    use_side = bool(kw.get("side"))
    hfe = HighBands().cuda()
    y, sr = sf.read(wav48, dtype="float32")
    assert sr == SR
    y = y.mean(1) if y.ndim > 1 else y
    if kw.get("ir"):                                      # FILE.npy:INDEX, a room the test sermon is played through
        path, idx = kw["ir"].rsplit(":", 1)
        y = fftconvolve(y, np.load(path)[int(idx)])[: len(y)].astype(np.float32)
    y16 = to16(y[: len(y) // 3 * 3])
    edges = cut_points(y16, 16000)
    n = int(kw.get("n", 0)) or len(edges) - 1
    with open(out, "x", encoding="utf-8") as fo:
        for i, (s, e) in enumerate(zip(edges[:n], edges[1:n + 1])):
            clip = loud(y[3 * s: 3 * e])
            t = time.time()
            text, ran_out = transcribe(proc, model, hfe, clip, use_side)
            fo.write(json.dumps({"window": i + 1, "start": round(s / 16000, 2), "end": round(e / 16000, 2),
                                 "side": kw.get("side"), "loud_ko": text, "ran_out": ran_out,
                                 "s": round(time.time() - t, 2)}, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"{i + 1}/{n} {time.time() - t:.1f}s", flush=True)


def cer(ref, hyp):
    """character error rate after dropping everything but letters and digits"""
    a = re.sub(r"[\W_]+", "", ref)
    b = re.sub(r"[\W_]+", "", hyp)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(1, len(a))


@torch.no_grad()
def labelcheck(model_dir, clips_path, out, n="40", seed="0"):
    """Do the training labels match the sound? Stock Seoul (no side path) transcribes a random sample of training
    clips, dry, scored against each clip's own subtitle text. On the held-out sermon stock Seoul is ~10% here."""
    proc, model, side = load(model_dir)
    hfe = HighBands().cuda()
    rows = [json.loads(l) for l in open(clips_path, encoding="utf-8")]
    sample = random.Random(int(seed)).sample(rows, int(n))
    with open(out, "x", encoding="utf-8") as fo:
        for i, r in enumerate(sample):
            text, ran_out = transcribe(proc, model, hfe, loud(read_clip(r)), False)
            fo.write(json.dumps({"sermon": r["sermon"], "t0": r["t0"], "t1": r["t1"], "label": r["text"], "heard": text,
                                 "cer": round(cer(r["text"], text), 4), "ran_out": ran_out}, ensure_ascii=False) + "\n")
            fo.flush()
            print(i + 1, r["sermon"], round(cer(r["text"], text), 3), flush=True)


@torch.no_grad()
def probe(model_dir, wav48, side_pt, irs_path, out, n="20"):
    """How hard the side path pushes: per window, the mean size of the old ears' rows and of the side's addition,
    and the new-band input's mean and spread. Dry, then through every clap in IRS (same windows, cut on the dry
    version, so the rooms line up)."""
    proc, model, side = load(model_dir, side_pt)
    hfe = HighBands().cuda()
    y, sr = sf.read(wav48, dtype="float32")
    assert sr == SR
    y = y.mean(1) if y.ndim > 1 else y
    y = y[: len(y) // 3 * 3]
    edges = cut_points(to16(y), 16000)[: int(n) + 1]
    rooms = [("dry", y)] + [(f"ir{k}", fftconvolve(y, ir)[: len(y)].astype(np.float32))
                            for k, ir in enumerate(np.load(irs_path))]
    with open(out, "x", encoding="utf-8") as fo:
        for name, yy in rooms:
            for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
                model._probe = []
                enc, _ = encode(proc, model, hfe, loud(yy[3 * s: 3 * e]), use_side=True)
                model(**enc)
                h, add, mu, sd = np.mean(model._probe, axis=0)
                fo.write(json.dumps({"room": name, "window": i + 1, "old_rows": round(h, 3), "side_add": round(add, 3),
                                     "ratio": round(add / h, 4), "hi_mean": round(mu, 3), "hi_std": round(sd, 3)}) + "\n")
            print(name, "done", flush=True)
    model._probe = None


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"clips": clips, "irs": irs, "check": check, "train": train, "asr": asr, "probe": probe, "labelcheck": labelcheck}[cmd](*args)
