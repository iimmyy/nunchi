# The true answer key for the voice generator: the file's own (8-bit) weights dequantized, run in 64-bit, on the
# inputs of a pass dump (LG_GEN_DUMP format). Then: how far each set of scores is from it.
#   python3 gen_exact.py GGUF_PY_DIR MMPROJ.gguf DUMP.bin N_PASSES NAME=SCORES.bin ...
import struct, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(sys.argv[2]); T = {t.name: t for t in r.tensors}
def W(n): t = T["a.gen.code." + n]; return dequantize(t.data, t.tensor_type).astype(np.float64)
def read(p, n_cb=12):
    out = []
    with open(p, "rb") as f:
        while True:
            b = f.read(4)
            if len(b) < 4: break
            kind = struct.unpack("i", b)[0]; ne = struct.unpack("i", f.read(4))[0]
            h = np.frombuffer(f.read(4 * ne), np.float32); codes = np.frombuffer(f.read(4 * n_cb), np.int32)
            no = struct.unpack("i", f.read(4))[0]; o = np.frombuffer(f.read(4 * no), np.float32)
            if kind == 0: out.append((h, codes, o))
    return out
recs = read(sys.argv[3])[: int(sys.argv[4])]
eps, theta = 1e-5, 1e6
embd = W("embd.weight")                 # [12, 2017, 1024]
mask, onorm, hnorm = W("mask_embd"), W("output_norm.weight"), W("head_norm.weight")   # hnorm [12, 1024]
pw, pb = W("proj_in.weight"), W("proj_in.bias")
L = []
i = 0
while f"a.gen.code.blk.{i}.ln1.weight" in T:
    p = f"blk.{i}."
    L.append({k: W(p + k + ".weight") for k in ["ln1", "ln2", "attn_q", "attn_k", "attn_v", "attn_out", "attn_gate", "attn_q_norm", "attn_k_norm", "ffn_gate", "ffn_up", "ffn_down"]}); i += 1
D = L[0]["attn_q_norm"].shape[0]; H = L[0]["attn_q"].shape[0] // D; Hk = L[0]["attn_k"].shape[0] // D; P = 12
def norm(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
def rope(v):
    half = D // 2; inv = theta ** (-np.arange(half) * 2.0 / D); ang = np.arange(P)[:, None] * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]; a, b = v[..., :half], v[..., half:]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)
truth = []
for h, codes, _ in recs:
    base = (pw @ h.astype(np.float64) + pb).reshape(P, -1)
    x = base + np.stack([embd[c, codes[c]] if codes[c] >= 0 else mask for c in range(P)])
    for l in L:
        hh = norm(x, l["ln1"])
        q = norm((hh @ l["attn_q"].T).reshape(P, H, D), l["attn_q_norm"]); k = norm((hh @ l["attn_k"].T).reshape(P, Hk, D), l["attn_k_norm"])
        v = (hh @ l["attn_v"].T).reshape(P, Hk, D); g = 1 / (1 + np.exp(-(hh @ l["attn_gate"].T)))
        q, k = rope(q), rope(k)
        k, v = np.repeat(k, H // Hk, 1), np.repeat(v, H // Hk, 1)
        s = np.einsum("phd,thd->hpt", q, k) / np.sqrt(D); s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        a = np.einsum("hpt,thd->phd", s, v) * g[:, :, None]
        x = x + a.reshape(P, -1) @ l["attn_out"].T
        hh = norm(x, l["ln2"]); gg = hh @ l["ffn_gate"].T
        x = x + (gg / (1 + np.exp(-gg)) * (hh @ l["ffn_up"].T)) @ l["ffn_down"].T
    z = norm(norm(x, onorm), hnorm)
    truth.append(np.concatenate([embd[c] @ z[c] for c in range(P)]))
print(f"{'scores':14s} {'error mean':>11s} {'max':>9s} {'top code same':>14s}")
for arg in [f"llama.cpp={sys.argv[3]}"] + sys.argv[5:]:
    name, p = arg.split("=", 1)
    rs = read(p)[: len(truth)]
    e = [np.linalg.norm(o - t) / np.linalg.norm(t) for (_, _, o), t in zip(rs, truth)]
    same = sum(int((o.reshape(P, -1).argmax(1) == t.reshape(P, -1).argmax(1)).sum()) for (_, _, o), t in zip(rs, truth))
    print(f"{name:14s} {np.mean(e):11.2e} {np.max(e):9.2e} {same:>8d}/{P * len(truth)}")
