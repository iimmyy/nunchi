"""Ogg for his memory room (Immy 2026-11-08: "just make it an ogg file"). The room a chat takes in his head is the
keys and values every own-room floor keeps for each word piece (the KV cache: E4B, floors 0-23; floors 24-41 reuse
floor 22's and 23's). This squeezes the old part of the chat the way an ogg file squeezes sound, without touching the
model:

  1. hearing model   what his look-ups actually pick up from a stored key (the queries of every floor that reads
                     it), and what a stored value passes on (that floor's output mixer, o_proj): attention-aware
                     distortion
  2. rotate          into the directions that matter most as heard (PCA after weighting by the hearing model), keys
                     with their position stamp (RoPE) taken off first and put back after
  3. spend bits      where it's heard: reverse water-filling over the directions, 0 to 8 bits each, so the average
                     hits the budget (avg 4 bits = 4x less room than 16-bit, avg 2 = 8x)
  4. quantize        each direction with the best uniform quantizer for its bit count (Gaussian, Max 1960)

Calibration reads public text only (the finder's FineWeb-Edu + UltraChat rows), never the test bank. This file only
measures how it sounds: the squeezed numbers are decoded right back, so answers show the quality the room saving
costs. Prior art this follows: KVTC (Staniszewski & Lancucki, ICLR 2026), SPECTRA and AATC (2026), KVQuant (pre-RoPE
keys).
"""
import math

import torch

STEP = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.335, 5: 0.188, 6: 0.104, 7: 0.057, 8: 0.031}   # x sigma, per bit count
MAX_BITS = 8
SINK = 4                  # the first word pieces stay untouched (attention sinks)
FAST_SPAN = 2048          # position-stamp pairs that turn a full circle within this many pieces average out


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def stamps(lm, n0, n1, layer_type, device):
    pos = torch.arange(n0, n1, device=device)[None]
    cos, sin = lm.rotary_emb(torch.zeros(1, device=device, dtype=torch.float32), pos, layer_type)
    return cos[0].float(), sin[0].float()


def readers(cfg):
    """own-room floor -> every floor that reads its keys and values (itself, plus the top floors that reuse it)"""
    n, first = cfg.num_hidden_layers, cfg.num_hidden_layers - cfg.num_kv_shared_layers
    t = cfg.layer_types
    last = {lt: max(i for i in range(first) if t[i] == lt) for lt in set(t[:first])}
    r = {i: [i] for i in range(first)}
    for j in range(first, n):
        r[last[t[j]]].append(j)
    return r


def _sqrtm(w, eps=1e-4):
    w = (w + w.T) / 2
    e, u = torch.linalg.eigh(w)
    e = e.clamp_min(eps * e.max())
    return (u * e.sqrt()) @ u.T, (u / e.sqrt()) @ u.T


@torch.no_grad()
def calibrate(model, rows, batch=4):
    """rows: int tensor [R, T] of public text. Returns the codec: per (floor, kv head, "k"/"v") the weighted
    rotation in and out, the mean, and each direction's loudness as heard"""
    lm = model.model.language_model
    cfg = lm.config
    rd = readers(cfg)
    own = sorted(rd)
    att = [layer.self_attn for layer in lm.layers]
    kvh = {i: att[i].k_proj.out_features // att[i].head_dim for i in own}
    stats, hooks = {}, []

    def acc(key, x):                                    # x [N, d] float64
        s = stats.setdefault(key, [0, 0.0, 0.0])
        s[0] += x.shape[0]
        s[1] = s[1] + x.sum(0)
        s[2] = s[2] + x.T @ x

    def grab(i, kind):
        def hook(mod, inp, out):                         # out [b, s, heads, d] (before the position stamp)
            x = out[:, SINK:].double()
            if kind == "q":
                g = x.shape[2] // kvh[rd_of[i]]
                for h in range(x.shape[2]):
                    acc((i, h // g, "q"), x[:, :, h].reshape(-1, x.shape[-1]))
            else:
                for h in range(x.shape[2]):
                    acc((i, h, kind), x[:, :, h].reshape(-1, x.shape[-1]))
        return hook

    rd_of = {j: i for i, js in rd.items() for j in js}
    for j, a in enumerate(att):
        hooks.append(a.q_norm.register_forward_hook(grab(j, "q")))
        if j in rd:
            hooks.append(a.k_norm.register_forward_hook(grab(j, "k")))
            hooks.append(a.v_norm.register_forward_hook(grab(j, "v")))
    try:
        for s in range(0, len(rows), batch):
            lm(input_ids=rows[s:s + batch].cuda())
    finally:
        for h in hooks:
            h.remove()

    codec = {}
    for i in own:
        d = att[i].head_dim
        lt = cfg.layer_types[i]
        inv = getattr(lm.rotary_emb, f"{lt}_inv_freq").double().cuda()
        fast = (inv * FAST_SPAN > 2 * math.pi).nonzero().flatten()
        for g in range(kvh[i]):
            for kind in ("k", "v"):
                n, s1, s2 = stats[(i, g, kind)]
                mu = s1 / n
                cov = s2 / n - torch.outer(mu, mu)
                if kind == "k":                          # heard through the queries of every floor reading it
                    w = sum(stats[(j, g, "q")][2] / stats[(j, g, "q")][0] for j in rd[i])
                    for p in fast.tolist():              # fast-turning stamp pairs: only their joint level is heard
                        q = p + d // 2
                        lvl = (w[p, p] + w[q, q]) / 2
                        w[p, :] = 0
                        w[:, p] = 0
                        w[q, :] = 0
                        w[:, q] = 0
                        w[p, p] = w[q, q] = lvl
                else:                                    # passed on through each reading floor's output mixer
                    w = 0
                    for j in rd[i]:
                        o = att[j].o_proj.weight.double()
                        grp = att[j].q_proj.out_features // att[j].head_dim // kvh[i]
                        for h in range(g * grp, (g + 1) * grp):
                            b = o[:, h * d:(h + 1) * d]
                            w = w + b.T @ b
                a, ainv = _sqrtm(w)
                lam, vec = torch.linalg.eigh(a @ cov @ a)
                lam, vec = lam.flip(0).clamp_min(0), vec.flip(1)
                codec[(i, g, kind)] = dict(enc=(a @ vec).float(), dec=(vec.T @ ainv).float(), mu=mu.float(),
                                           lam=lam.float())
    return codec


def allocate(lam, avg_bits):
    """reverse water-filling: bits per direction, 0..8, averaging avg_bits"""
    lam = lam.double().clamp_min(1e-30)
    lo, hi = math.log2(float(lam.min())) - 20, math.log2(float(lam.max())) + 2
    bits = None
    for _ in range(80):
        mid = (lo + hi) / 2
        bits = (0.5 * (torch.log2(lam) - mid)).round().clamp(0, MAX_BITS)
        if bits.mean() > avg_bits:
            lo = mid
        else:
            hi = mid
    return bits.int()


def squeeze(x, c, bits):
    """x [N, d] float -> the same after coding at these bits per direction and decoding"""
    coef = (x - c["mu"]) @ c["enc"]
    sigma = c["lam"].sqrt()
    out = torch.zeros_like(coef)
    for b in range(1, MAX_BITS + 1):
        idx = (bits == b).nonzero().flatten()
        if len(idx):
            step = STEP[b] * sigma[idx]
            q = torch.floor(coef[:, idx] / step).clamp(-2 ** (b - 1), 2 ** (b - 1) - 1)
            out[:, idx] = (q + 0.5) * step
    return out @ c["dec"] + c["mu"]


def plan(codec, avg_bits):
    return {k: allocate(c["lam"], avg_bits).to(c["lam"].device) for k, c in codec.items()}


@torch.no_grad()
def squeeze_cache(model, cache, start, end, codec, bits):
    """squeeze the stored keys and values of word pieces start..end-1 on every own-room floor, in place"""
    lm = model.model.language_model
    cfg = lm.config
    st = {}
    for (i, g, kind), c in codec.items():
        layer = cache.layers[i]
        t = layer.keys if kind == "k" else layer.values             # [1, kv heads, pieces, d]
        x = t[0, g, start:end].float()
        if kind == "k":
            lt = cfg.layer_types[i]
            if lt not in st:
                st[lt] = stamps(lm, start, end, lt, x.device)
            cos, sin = st[lt]
            assert cos.shape[-1] == x.shape[-1]
            x = (x * cos - rotate_half(x) * sin) / (cos * cos + sin * sin)
            y = squeeze(x, c, bits[(i, g, kind)])
            y = y * cos + rotate_half(y) * sin
        else:
            y = squeeze(x, c, bits[(i, g, kind)])
        t[0, g, start:end] = y.to(t.dtype)
