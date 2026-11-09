# The true answer key for a 4-bit Seoul file: the file's own weights, dequantized, run in 64-bit with no input
# rounding. llama.cpp and imi both round inputs to 8-bit inside every table read; this measures how far each is
# from the model the file actually describes.
#   python3 ref_exact.py GGUF_PY_DIR MODEL.gguf REF.bin N_POS OUT.bin
# REF.bin gives the token list (imi answer-key format). OUT.bin is written in the same format.
import struct, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
from gguf.quants import dequantize

model, ref, npos, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
r = GGUFReader(model)
T = {t.name: t for t in r.tensors}
def W(name): t = T[name]; return dequantize(t.data, t.tensor_type).astype(np.float64)
def key(name, default=None):
    f = r.fields.get(name)
    if f is None: return default
    v = f.parts[f.data[0]] if len(f.data) == 1 else [f.parts[i][0] for i in f.data]
    return v[0] if hasattr(v, "__len__") and len(f.data) == 1 else v

eps = float(key("lfm3.attention.layer_norm_rms_epsilon", 1e-5))
base, base_swa = float(key("lfm3.rope.freq_base", 1e6)), float(key("lfm3.rope.freq_base_swa", 1e4))
n_swa = int(key("lfm3.attention.sliding_window", 0))
n_layer = int(key("lfm3.block_count"))
swa_pat = [bool(x) for x in key("lfm3.attention.sliding_window_pattern")]
H, D = int(key("lfm3.attention.head_count", 16)), int(key("lfm3.attention.key_length", 128))
ff = W("rope_freqs.weight") if "rope_freqs.weight" in T else None

with open(ref, "rb") as f:
    ne, nv, n = struct.unpack("3i", f.read(12))
    toks = []
    for i in range(n):
        toks.append(struct.unpack("i", f.read(4))[0]); f.seek(4 * (ne + nv), 1)
toks = toks[:npos]
Tn = len(toks)
emb = W("token_embd.weight")
X = emb[toks]                                            # [T, E]

def norm(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
def rope(v, b, fr):                                      # v [T, heads, D], NEOX
    half = D // 2
    inv = b ** (-np.arange(half) * 2.0 / D)
    if fr is not None: inv = inv / fr
    ang = np.arange(Tn)[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = v[..., :half], v[..., half:]
    return np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], -1)

for il in range(n_layer):
    p = f"blk.{il}."
    h = norm(X, W(p + "attn_norm.weight"))
    if p + "shortconv.conv.weight" in T:
        E = X.shape[1]
        bcx = h @ W(p + "shortconv.in_proj.weight").T
        b, c, xx = bcx[:, :E], bcx[:, E:2 * E], bcx[:, 2 * E:]
        cw = W(p + "shortconv.conv.weight")             # [E, K]
        K = cw.shape[1]
        bx = np.concatenate([np.zeros((K - 1, E)), b * xx], 0)
        co = np.stack([(bx[t:t + K] * cw.T).sum(0) for t in range(Tn)])
        y = (c * co) @ W(p + "shortconv.out_proj.weight").T
    else:
        q = (h @ W(p + "attn_q.weight").T).reshape(Tn, H, D)
        k = h @ W(p + "attn_k.weight").T; Hk = k.shape[1] // D; k = k.reshape(Tn, Hk, D)
        v = (h @ W(p + "attn_v.weight").T).reshape(Tn, Hk, D)
        g = 1 / (1 + np.exp(-(h @ W(p + "attn_gate.weight").T)))   # [T, H]
        q, k = norm(q, W(p + "attn_q_norm.weight")), norm(k, W(p + "attn_k_norm.weight"))
        sw = swa_pat[il]
        q, k = rope(q, base_swa if sw else base, None if sw else ff), rope(k, base_swa if sw else base, None if sw else ff)
        a = np.zeros((Tn, H, D))
        for t in range(Tn):
            s0 = max(0, t - n_swa + 1) if (sw and n_swa > 0) else 0
            for hh in range(H):
                kv = hh // (H // Hk)
                sc = k[s0:t + 1, kv] @ q[t, hh] / np.sqrt(D)
                pr = np.exp(sc - sc.max()); pr /= pr.sum()
                a[t, hh] = pr @ v[s0:t + 1, kv]
        a *= g[:, :, None]
        y = a.reshape(Tn, H * D) @ W(p + "attn_output.weight").T
    X = X + norm(y, W(p + "post_attention_norm.weight"))
    h = norm(X, W(p + "ffn_norm.weight"))
    gg, uu = h @ W(p + "ffn_gate.weight").T, h @ W(p + "ffn_up.weight").T
    fo = (gg / (1 + np.exp(-gg)) * uu) @ W(p + "ffn_down.weight").T
    X = X + norm(fo, W(p + "post_ffw_norm.weight"))
    print("layer", il, flush=True)
hid = norm(X, W("output_norm.weight"))
lg = hid @ emb.T
with open(out, "wb") as f:
    f.write(struct.pack("3i", hid.shape[1], lg.shape[1], Tn))
    for t in range(Tn):
        f.write(struct.pack("i", toks[t])); f.write(hid[t].astype(np.float32).tobytes()); f.write(lg[t].astype(np.float32).tobytes())
print("wrote", out, Tn, "positions")
