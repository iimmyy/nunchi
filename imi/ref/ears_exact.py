# The true ears output for a 16 kHz clip: the ears file's own weights in 64-bit, the mel with an exact FFT, the whole
# clip at once (a zero stream state is the same as starting at the clip's first sample).
#   python3 ears_exact.py GGUF_PY_DIR MMPROJ.gguf CLIP.f32 OUT.bin [MEL.bin]
# CLIP.f32: raw float32 samples at 16 kHz. OUT.bin: [int32 n_pos][int32 2048][float32 data], the layout the fork's
# MTMD_DEBUG_EMBEDDINGS dump uses. MEL.bin: [int32 n_frames][int32 128][float32 data], the encoder's input.
import math, struct, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(sys.argv[2]); T = {t.name: t for t in r.tensors}
def W(n): t = T[n]; return dequantize(t.data, t.tensor_type).astype(np.float64)
def save(path, a):
    with open(path, "wb") as f:
        f.write(struct.pack("ii", *a.shape)); f.write(a.astype("<f4").tobytes())

# ---- mel: 400-sample periodic Hann frames every 160 samples, power spectrum, 128 Slaney mels, log10, (v + 4) / 4
pcm = np.fromfile(sys.argv[3], "<f4").astype(np.float64)
win, hop, sr, n_mel = 400, 160, 16000, 128
n = len(pcm)
n_pad = win if n <= win else n + (hop - (n - win) % hop) % hop
y = np.zeros(n_pad); y[:n] = pcm
n_fr = (n_pad - win) // hop + 1
hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win) / win))
spec = np.abs(np.fft.rfft(np.stack([y[i * hop:i * hop + win] * hann for i in range(n_fr)]), win)) ** 2
lin, min_log_mel, step = 3 / 200, 15.0, math.log(6.4) / 27
def hz2mel(f): return f * lin if f < 1000 else min_log_mel + math.log(f / 1000) / step
def mel2hz(m): return m / lin if m < min_log_mel else 1000 * math.exp((m - min_log_mel) * step)
lo, hi = hz2mel(0.0), hz2mel(sr / 2)
pts = [mel2hz(lo + (hi - lo) * i / (n_mel + 1)) for i in range(n_mel + 2)]
freqs = np.arange(win // 2 + 1) * sr / win
fb = np.zeros((n_mel, win // 2 + 1))
for m in range(n_mel):
    fl, fc, fr = pts[m:m + 3]
    up = (freqs >= fl) & (freqs <= fc); down = (freqs > fc) & (freqs <= fr)
    fb[m, up] = (freqs[up] - fl) / (fc - fl); fb[m, down] = (fr - freqs[down]) / (fr - fc)
    fb[m] *= 2 / (fr - fl)
mel = (np.log10(np.maximum(spec @ fb.T, 1e-12)) + 4) / 4          # [n_fr, 128]
if len(sys.argv) > 5: save(sys.argv[5], mel)

# ---- downsampler: three stride-2 3x3 convs over [time, freq], zero edges (the stream's left column starts at zero)
def conv_s2(a, w, b, depthwise):   # a [T][F][C]; w [oc][ic][kh][kw], kw along time
    t_n, f_n, c_n = a.shape
    to, fo = (t_n + 1) // 2, (f_n + 1) // 2
    p = np.zeros((t_n + 2, f_n + 2, c_n)); p[1:t_n + 1, 1:f_n + 1] = a
    out = np.zeros((to, fo, w.shape[0]))
    for kh in range(3):
        for kw in range(3):
            patch = p[kw:kw + 2 * to:2, kh:kh + 2 * fo:2]
            out += patch * w[:, 0, kh, kw] if depthwise else patch[..., :1] * w[:, 0, kh, kw]
    return out + b.reshape(-1)
def silu(v): return v / (1 + np.exp(-v))
def pw(a, w, b): return a @ w[:, :, 0, 0].T + b.reshape(-1)
a = silu(conv_s2(mel[:, :, None], W("a.conv1d.0.weight"), W("a.conv1d.0.bias"), False))
a = silu(pw(conv_s2(a, W("a.conv1d.2.weight"), W("a.conv1d.2.bias"), True), W("a.conv1d.3.weight"), W("a.conv1d.3.bias")))
a = silu(pw(conv_s2(a, W("a.conv1d.5.weight"), W("a.conv1d.5.bias"), True), W("a.conv1d.6.weight"), W("a.conv1d.6.bias")))
N = a.shape[0]
x = a.transpose(0, 2, 1).reshape(N, -1) @ W("a.pre_encode.out.weight").T + W("a.pre_encode.out.bias")   # feature = ch * 16 + freq

# ---- 17 causal LFM layers (short conv or windowed GQA attention), pre-norm only
eps, theta, window = 1e-5, 1e6, int(r.fields["clip.audio.window_size"].parts[-1][0])
def norm(v, w): return v / np.sqrt((v * v).mean(-1, keepdims=True) + eps) * w
def rope(v, D):
    half = D // 2; inv = theta ** (-np.arange(half) * 2.0 / D); ang = np.arange(N)[:, None] * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]; p0, p1 = v[..., :half], v[..., half:]
    return np.concatenate([p0 * c - p1 * s, p0 * s + p1 * c], -1)
il = 0
while f"a.blk.{il}.ln1.weight" in T:
    p = f"a.blk.{il}."
    h = norm(x, W(p + "ln1.weight")); E = x.shape[1]
    if p + "conv_dw.weight" in T:
        bcx = h @ W(p + "conv_pw1.weight").T
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
        att = np.zeros((N, H, D))
        for t in range(N):
            s0 = max(0, t - window + 1)
            for hh in range(H):
                kv = hh // (H // Hk)
                sc = k[s0:t + 1, kv] @ q[t, hh] / np.sqrt(D); pr = np.exp(sc - sc.max()); pr /= pr.sum()
                att[t, hh] = pr @ v[s0:t + 1, kv]
        y = att.reshape(N, -1) @ W(p + "attn_out.weight").T
    x = x + y
    h = norm(x, W(p + "ln2.weight")); g = h @ W(p + "ffn_gate.weight").T
    x = x + (silu(g) * (h @ W(p + "ffn_up.weight").T)) @ W(p + "ffn_down.weight").T
    il += 1

# ---- last norm, then the projector: LayerNorm -> 768 x 2048 -> GELU (erf) -> 2048 x 2048
x = norm(x, W("a.post_ln.weight"))
mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
x = (x - mu) / np.sqrt(var + 1e-5) * W("mm.a.mlp.0.weight") + W("mm.a.mlp.0.bias")
x = x @ W("mm.a.mlp.1.weight").T + W("mm.a.mlp.1.bias")
erf = np.vectorize(math.erf)
x = 0.5 * x * (1 + erf(x / math.sqrt(2)))
x = x @ W("mm.a.mlp.3.weight").T + W("mm.a.mlp.3.bias")
save(sys.argv[4], x)
print(f"ears truth: {n} samples -> {n_fr} mel frames -> {N} positions x {x.shape[1]}, {il} layers")
