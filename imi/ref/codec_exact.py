# The true sound for a pass dump's final codes: the codec file's own weights, dequantized, in 64-bit; the inverse
# STFT as the reference does it. Then the SNR of each WAV against it.
#   python3 codec_exact.py GGUF_PY_DIR MMPROJ.gguf DUMP.bin NAME=FILE.wav ...
import struct, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(sys.argv[2]); T = {t.name: t for t in r.tensors}
def W(n): t = T["a.gen.wav." + n]; return dequantize(t.data, t.tensor_type).astype(np.float64)
frames = []
with open(sys.argv[3], "rb") as f:
    while True:
        b = f.read(4)
        if len(b) < 4: break
        kind = struct.unpack("i", b)[0]; ne = struct.unpack("i", f.read(4))[0]; f.seek(4 * ne, 1)
        codes = np.frombuffer(f.read(48), np.int32).copy(); no = struct.unpack("i", f.read(4))[0]; f.seek(4 * no, 1)
        if kind == 1: frames.append(codes)
codes = np.array(frames)                       # [F, 12]
F, P, eps, win, theta = len(codes), 6, 1e-5, 30, 1e6
lvl = W("quant.first.codebook.weight")         # [2016, 4]
x = lvl[codes].reshape(F, -1) @ W("quant.first.out_proj.weight").T + W("quant.first.out_proj.bias")
x = x @ W("tfm.in_proj.weight").T + W("tfm.in_proj.bias")
x = np.repeat(x, P, 0)                         # [N, 512], N = 6F positions
N = len(x)
def norm(v, w): return v / np.sqrt((v * v).mean(-1, keepdims=True) + eps) * w
def rope(v, D):
    half = D // 2; inv = theta ** (-np.arange(half) * 2.0 / D); ang = np.arange(N)[:, None] * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]; a, b = v[..., :half], v[..., half:]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)
il = 0
while f"a.gen.wav.tfm.blk.{il}.ln1.weight" in T:
    p = f"tfm.blk.{il}."
    h = norm(x, W(p + "ln1.weight"))
    if f"a.gen.wav.{p}conv_dw.weight" in T:
        bcx = h @ W(p + "conv_pw1.weight").T; E = x.shape[1]
        b, c, xx = bcx[:, :E], bcx[:, E:2 * E], bcx[:, 2 * E:]
        cw = W(p + "conv_dw.weight"); K = cw.shape[1]
        bx = np.concatenate([np.zeros((K - 1, E)), b * xx])
        co = np.stack([(bx[t:t + K] * cw.T).sum(0) for t in range(N)])
        y = (c * co) @ W(p + "conv_pw2.weight").T
    else:
        qn = W(p + "attn_q_norm.weight"); D = qn.shape[0]
        q = (h @ W(p + "attn_q.weight").T).reshape(N, -1, D); k = (h @ W(p + "attn_k.weight").T).reshape(N, -1, D)
        v = (h @ W(p + "attn_v.weight").T).reshape(N, -1, D); H, Hk = q.shape[1], k.shape[1]
        q, k = rope(norm(q, qn), D), rope(norm(k, W(p + "attn_k_norm.weight")), D)
        g = 1 / (1 + np.exp(-(h @ W(p + "attn_gate.weight").T)))
        a = np.zeros((N, H, D))
        for t in range(N):
            s0 = max(0, t - win + 1)
            for hh in range(H):
                kv = hh // (H // Hk)
                sc = k[s0:t + 1, kv] @ q[t, hh] / np.sqrt(D); pr = np.exp(sc - sc.max()); pr /= pr.sum()
                a[t, hh] = pr @ v[s0:t + 1, kv]
        y = (a * g[:, :, None]).reshape(N, -1) @ W(p + "attn_out.weight").T
    x = x + norm(y, W(p + "attn_post_norm.weight"))
    h = norm(x, W(p + "ln2.weight")); gg = h @ W(p + "ffn_gate.weight").T
    x = x + norm((gg / (1 + np.exp(-gg)) * (h @ W(p + "ffn_up.weight").T)) @ W(p + "ffn_down.weight").T, W(p + "ffn_post_norm.weight"))
    il += 1
out = norm(x, W("tfm.output_norm.weight")) @ W("tfm.out_proj.weight").T + W("tfm.out_proj.bias")
nb = out.shape[1] // 2
spec = np.exp(out[:, :nb]) * np.exp(1j * out[:, nb:])
n_fft, hop = 1280, 320
hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
frames_t = np.fft.irfft(spec, n_fft, axis=1) * hann
total = (N - 1) * hop + n_fft
ola, ws = np.zeros(total), np.zeros(total)
for i in range(N): ola[i * hop:i * hop + n_fft] += frames_t[i]; ws[i * hop:i * hop + n_fft] += hann ** 2
pcm = np.where(ws > 1e-8, ola / np.maximum(ws, 1e-30), ola)
pad = (n_fft - hop) // 2
pcm = pcm[pad: pad + N * hop]                  # the reference trims (n_fft - hop) / 2 at both ends
print(f"truth: {len(pcm)} samples")
for arg in sys.argv[4:]:
    name, p = arg.split("=", 1)
    w = np.fromfile(p, np.int16, offset=44).astype(np.float64) / 32767.0
    n = min(len(w), len(pcm)); d = w[:n] - pcm[:n]
    print(f"{name:12s} SNR vs truth {10 * np.log10((pcm[:n] ** 2).sum() / (d ** 2).sum()):6.1f} dB  ({len(w)} samples)")
q = np.round(np.clip(pcm, -1, 1) * 32767) / 32767; d = q - pcm
print(f"{'(16-bit cut)':12s} SNR vs truth {10 * np.log10((pcm ** 2).sum() / (d ** 2).sum()):6.1f} dB  (the WAV format's own limit)")
