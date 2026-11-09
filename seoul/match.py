"""Output matching for stock llama.cpp types: keep every block scale of a stock-quantized matrix exactly as it is and
re-pick only the integer codes, column by column, so the matrix's output on real inputs matches the original
(GPTQ-style error compensation with a fixed grid). The result is byte-compatible with the stock type.

  match.py test  <ref.gguf> <donor.gguf> <act dir> <tensor>   one matrix: output error, stock rounding vs matched
  match.py build <ref.gguf> <donor.gguf> <act dir> <out.gguf> every matrix with recorded inputs, matched

ref: the full-precision model (bf16). donor: the whole model at one stock type (same importance matrix as usual).
act dir: input rows recorded by llama-imatrix with LG_ACT_DUMP (one file per distinct input, f16).
Supported types: Q4_0, Q3_K, Q2_K.
"""
import json, os, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py")))
import gguf
from gguf.quants import dequantize

T = gguf.GGMLQuantizationType


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ grids: value = a * code + b, per element

def grid(raw, qt, n_col):
    """raw block bytes (rows, row bytes) -> per-element (a, b, code) arrays of shape (rows, n_col), codes in 0..qmax"""
    R = raw.shape[0]
    if qt == T.Q4_0:
        blk = raw.reshape(R, -1, 18)
        d = blk[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float32)
        qs = blk[:, :, 2:]
        code = np.concatenate([qs & 15, qs >> 4], axis=2)                       # element j and j+16
        a = np.repeat(d, 32, axis=1)
        return a, -8 * a, code.reshape(R, n_col).astype(np.int16), 15
    if qt == T.Q2_K:
        blk = raw.reshape(R, -1, 84)
        sc, qs = blk[:, :, :16], blk[:, :, 16:80]
        d = blk[:, :, 80:82].copy().view(np.float16)[:, :, 0].astype(np.float32)
        dm = blk[:, :, 82:84].copy().view(np.float16)[:, :, 0].astype(np.float32)
        code, a, b = _k_layout(R, qs, n_col)
        idx = _k_scale_index()                                                   # (256,) -> scale slot
        a[:] = (d[:, :, None] * (sc[:, :, idx] & 15)).reshape(R, n_col)
        b[:] = (-dm[:, :, None] * (sc[:, :, idx] >> 4)).reshape(R, n_col)
        return a, b, code, 3
    if qt == T.Q3_K:
        blk = raw.reshape(R, -1, 110)
        hm, qs, scb = blk[:, :, :32], blk[:, :, 32:96], blk[:, :, 96:108]
        d = blk[:, :, 108:110].copy().view(np.float16)[:, :, 0].astype(np.float32)
        low, a, b = _k_layout(R, qs, n_col)
        hi = _k_hbits(hm).reshape(R, n_col)
        s = _q3k_scales(scb)                                                     # (R, nb, 16) in -32..31
        a[:] = (d[:, :, None] * s[:, :, _k_scale_index()]).reshape(R, n_col)
        b[:] = -4 * a
        return a, b, (low + 4 * hi).astype(np.int16), 7
    raise SystemExit(f"type {qt.name} not supported")


def _k_pos():
    """element position in a 256 super-block for (n, j, h, l): n*128 + j*32 + h*16 + l"""
    n, j, h, l = np.meshgrid(np.arange(2), np.arange(4), np.arange(2), np.arange(16), indexing="ij")
    return n, j, h, l, (n * 128 + j * 32 + h * 16 + l)


def _k_scale_index():
    n, j, h, l, pos = _k_pos()
    idx = np.empty(256, dtype=np.int64)
    idx[pos.ravel()] = (n * 8 + j * 2 + h).ravel()
    return idx


def _k_layout(R, qs, n_col):
    """2-bit low codes of Q2_K / Q3_K: element (n,j,h,l) lives in qs[n*32 + h*16 + l] at bit shift 2j"""
    n, j, h, l, pos = _k_pos()
    byte = (n * 32 + h * 16 + l).ravel(); shift = (2 * j).ravel()
    code = np.empty((R, qs.shape[1], 256), dtype=np.int16)
    code[:, :, pos.ravel()] = (qs[:, :, byte] >> shift) & 3
    return code.reshape(R, n_col), np.empty((R, n_col), np.float32), np.empty((R, n_col), np.float32)


def _k_hbits(hm):
    """Q3_K high bit of element (n,j,h,l): hmask[h*16 + l] bit (n*4 + j)"""
    n, j, h, l, pos = _k_pos()
    out = np.empty(hm.shape[:2] + (256,), dtype=np.int16)
    out[:, :, pos.ravel()] = (hm[:, :, (h * 16 + l).ravel()] >> (n * 4 + j).ravel()) & 1
    return out


def _q3k_scales(scb):
    """12 bytes -> 16 signed 6-bit scales (ggml's Q3_K packing), minus 32"""
    aux = scb.copy().view(np.uint32)                                             # (R, nb, 3)
    a0, a1, tmp = aux[..., 0], aux[..., 1], aux[..., 2]
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    w = [(a0 & k2) | (((tmp >> 0) & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4),
         ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)]
    return np.stack(w, axis=-1).view(np.int8).astype(np.int32) - 32              # (R, nb, 16)


def pack(raw, qt, code):
    """write new codes into a copy of the stock block bytes (scales untouched)"""
    R = raw.shape[0]
    out = raw.copy()
    if qt == T.Q4_0:
        c = code.reshape(R, -1, 32).astype(np.uint8)
        out.reshape(R, -1, 18)[:, :, 2:] = c[:, :, :16] | (c[:, :, 16:] << 4)
        return out
    n, j, h, l, pos = _k_pos()
    nb = code.shape[1] // 256
    c = code.reshape(R, nb, 256)[:, :, pos.ravel()].reshape(R, nb, 2, 4, 2, 16).astype(np.uint8)  # n j h l
    low = c & 3
    qs = np.zeros((R, nb, 2, 2, 16), np.uint8)                                   # n h l
    for jj in range(4):
        qs |= low[:, :, :, jj] << (2 * jj)
    if qt == T.Q2_K:
        out.reshape(R, nb, 84)[:, :, 16:80] = qs.reshape(R, nb, 64)
        return out
    hm = np.zeros((R, nb, 2, 16), np.uint8)                                      # h l
    for nn in range(2):
        for jj in range(4):
            hm |= ((c[:, :, nn, jj] >> 2) & 1) << (nn * 4 + jj)
    blk = out.reshape(R, nb, 110)
    blk[:, :, :32] = hm.reshape(R, nb, 32)
    blk[:, :, 32:96] = qs.reshape(R, nb, 64)
    return out


# ------------------------------------------------------------------ matching

ACT = {"attn_q": "attn_q", "attn_k": "attn_q", "attn_v": "attn_q", "ffn_gate": "ffn_gate", "ffn_up": "ffn_gate",
       "attn_output": "attn_output", "ffn_down": "ffn_down",
       # LFM3 (Seoul): the attention's output gate reads the same input as q; the short-conv layers' two matrices
       "attn_gate": "attn_q", "shortconv.in_proj": "shortconv.in_proj", "shortconv.out_proj": "shortconv.out_proj"}


HOLD = 8   # without a hold.json: every HOLD-th recorded row is held out of the fit and only used to check the result


def hessian(act_dir, name):
    m = re.match(r"(blk\.\d+)\.([a-z_.]+)\.weight$", name)
    f = Path(act_dir) / f"{m.group(1)}.{ACT[m.group(2)]}.weight.f16"
    return f if f.exists() else None


def rows(f, n_col, step=8192):
    """recorded input rows in order, in pieces (a full recording can be several GB)"""
    x = np.memmap(f, dtype=np.float16, mode="r").reshape(-1, n_col)
    for i in range(0, len(x), step):
        yield i, np.asarray(x[i:i + step], dtype=np.float32)


def hold_from(f):
    """row where the held-out documents start (calib_split.json saved as hold.json next to the recording), or None"""
    h = Path(f).parent / "hold.json"
    return json.loads(h.read_text())["hold_from_row"] if h.exists() else None


def grams(f, n_col, fq=None):
    """X^T X / n over the fit rows and over the held-out rows: whole held-out documents when the recording has a
    hold.json (every row from hold_from_row on), else every HOLD-th row. With fq (inputs recorded from the partly
    compressed model, same text and rows): Hq = Xq^T Xq / n and C = X^T Xq / n on the fit rows, for the sequential target."""
    h0 = hold_from(f)
    keys = ("fit", "hold") if fq is None else ("fit", "hold", "C", "C_hold", "X_hold")
    acc = {k: np.zeros((n_col, n_col)) for k in keys}
    cnt = {"fit": 0, "hold": 0}
    src = rows(f, n_col) if fq is None else zip(rows(f, n_col), rows(fq, n_col))
    for item in src:
        (i, X), Xq = (item, None) if fq is None else (item[0], item[1][1])
        n = len(X) if Xq is None else min(len(X), len(Xq))
        idx = np.arange(i, i + n)
        hold = idx >= h0 if h0 is not None else (idx % HOLD) == HOLD - 1
        A = (X if Xq is None else Xq)[:n]                           # the inputs this matrix will actually see
        acc["fit"] += A[~hold].T @ A[~hold]                         # each piece in float32 (fast), summed in float64
        acc["hold"] += A[hold].T @ A[hold]
        if Xq is not None:
            acc["C"] += X[:n][~hold].T @ A[~hold]
            acc["C_hold"] += X[:n][hold].T @ A[hold]                # for the true held-out error vs the original
            acc["X_hold"] += X[:n][hold].T @ X[:n][hold]
        cnt["fit"] += int((~hold).sum()); cnt["hold"] += int(hold.sum())
    out = {k: v / max(cnt["hold" if "hold" in k else "fit"], 1) for k, v in acc.items()}
    out["n_fit"], out["n_hold"] = cnt["fit"], cnt["hold"]
    return out


def out_err(E, H):
    return float(np.einsum("ij,ij->", E @ H, E))


def match(W, H, a, b, code0, qmax, damp, block=128, order=None):
    """GPTQ with a fixed grid: W (rows, cols) float, H (cols, cols); returns new codes. order="act" visits the columns
    with the most input energy first (GPTQ act-order); the grid is per element, so any column order is exact."""
    if order == "act":
        p = np.argsort(-np.diag(H), kind="stable"); inv = np.argsort(p)
        return match(W[:, p], H[np.ix_(p, p)], a[:, p], b[:, p], code0[:, p], qmax, damp, block)[:, inv]
    dt = np.float64 if W.shape[1] <= 12288 else np.float32   # 31B ffn_down (21504 inputs): float32 halves memory
    W = W.astype(dt).copy()
    H = H.astype(dt).copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H[np.diag_indices_from(H)] += damp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T                                # upper
    a = a.astype(dt); b = b.astype(dt)
    code = code0.copy()
    C = W.shape[1]
    for i1 in range(0, C, block):
        i2 = min(i1 + block, C)
        W1 = W[:, i1:i2].copy()
        E1 = np.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for k in range(i2 - i1):
            col = i1 + k
            w, ac, bc = W1[:, k], a[:, col], b[:, col]
            ok = ac != 0
            c = code0[:, col].astype(dt)
            c[ok] = np.clip(np.rint((w[ok] - bc[ok]) / ac[ok]), 0, qmax)
            q = ac * c + bc
            e = (w - q) / Hi[k, k]
            W1[:, k:] -= np.outer(e, Hi[k, k:])
            E1[:, k] = e
            code[:, col] = c
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return code


def _q3k_pack_scales(u):
    """16 unsigned 6-bit scales per super-block (R, nb, 16) -> ggml's 12-byte Q3_K packing (R, nb, 12)"""
    u = u.astype(np.uint8)
    aux0 = (u[..., 0:4] & 15) | ((u[..., 8:12] & 15) << 4)
    aux1 = (u[..., 4:8] & 15) | ((u[..., 12:16] & 15) << 4)
    tmp = (u[..., 0:4] >> 4) | ((u[..., 4:8] >> 4) << 2) | ((u[..., 8:12] >> 4) << 4) | ((u[..., 12:16] >> 4) << 6)
    return np.concatenate([aux0, aux1, tmp], axis=-1).astype(np.uint8)


def refit(raw, qt, W, code, wt):
    """re-fit every block scale to the codes the matcher chose (least squares weighted by the input energy wt per
    column), re-encode it exactly in the format; returns new block bytes"""
    R, C = W.shape
    out = raw.copy()
    W = W.astype(np.float64); wt = np.broadcast_to(wt.astype(np.float64), W.shape)
    if qt == T.Q4_0:                                                             # value = d * (c - 8), 32 per block
        cc = code.reshape(R, -1, 32).astype(np.float64) - 8; ww = W.reshape(R, -1, 32); w8 = wt.reshape(R, -1, 32)
        num, den = (w8 * ww * cc).sum(2), (w8 * cc * cc).sum(2)
        old = out.reshape(R, -1, 18)[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float64)
        d = np.where(den > 0, num / np.where(den > 0, den, 1), old)
        out.reshape(R, -1, 18)[:, :, :2] = d.astype(np.float16)[:, :, None].view(np.uint8)
        return out
    n, j, h, l, pos = _k_pos()
    nb = C // 256
    perm = lambda x: x.reshape(R, nb, 256)                                        # element order inside the super-block
    sub = _k_scale_index()                                                        # element -> sub-block (16 of 16)
    cc, ww, w8 = perm(code).astype(np.float64), perm(W), perm(wt)
    S = np.zeros((R, nb, 16, 3, 3))                                               # weighted normal equations per sub-block
    for k, s in enumerate(range(16)):
        m = sub == s
        c_, w_, t_ = cc[:, :, m], ww[:, :, m], w8[:, :, m]
        S[:, :, s, 0, 0] = (t_ * c_ * c_).sum(2); S[:, :, s, 0, 1] = (t_ * c_).sum(2); S[:, :, s, 1, 1] = t_.sum(2)
        S[:, :, s, 0, 2] = (t_ * c_ * w_).sum(2); S[:, :, s, 1, 2] = (t_ * w_).sum(2)
    if qt == T.Q3_K:                                                              # value = a * (c - 4), one a per sub-block
        num = S[..., 0, 2] - 4 * S[..., 1, 2]
        den = S[..., 0, 0] - 8 * S[..., 0, 1] + 16 * S[..., 1, 1]
        blk = out.reshape(R, nb, 110)
        d_old = blk[:, :, 108:110].copy().view(np.float16)[:, :, 0].astype(np.float64)
        a_old = d_old[:, :, None] * _q3k_scales(blk[:, :, 96:108])
        a = np.where(den > 0, num / np.where(den > 0, den, 1), a_old)
        amax = np.abs(a).max(2)
        d = np.where(amax > 0, amax / 31.5, d_old).astype(np.float16).astype(np.float64)
        s = np.clip(np.rint(a / np.where(d[:, :, None] == 0, 1, d[:, :, None])), -32, 31) + 32
        blk[:, :, 96:108] = _q3k_pack_scales(s)
        blk[:, :, 108:110] = d.astype(np.float16)[:, :, None].view(np.uint8)
        return out
    if qt == T.Q2_K:                                                              # value = a * c + b, a = d*sc, b = -dmin*m
        det = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] ** 2
        ok = det > 1e-30
        a = np.where(ok, (S[..., 0, 2] * S[..., 1, 1] - S[..., 0, 1] * S[..., 1, 2]) / np.where(ok, det, 1), np.nan)
        b = np.where(ok, (S[..., 0, 0] * S[..., 1, 2] - S[..., 0, 1] * S[..., 0, 2]) / np.where(ok, det, 1), np.nan)
        blk = out.reshape(R, nb, 84)
        sc_old = blk[:, :, :16]
        d_old = blk[:, :, 80:82].copy().view(np.float16)[:, :, 0].astype(np.float64)
        m_old = blk[:, :, 82:84].copy().view(np.float16)[:, :, 0].astype(np.float64)
        a = np.where(np.isnan(a), d_old[:, :, None] * (sc_old & 15), np.maximum(a, 0))
        b = np.where(np.isnan(b), -m_old[:, :, None] * (sc_old >> 4), np.minimum(b, 0))
        amax, bmax = a.max(2), (-b).max(2)
        d = np.where(amax > 0, amax / 15, d_old).astype(np.float16).astype(np.float64)
        dm = np.where(bmax > 0, bmax / 15, m_old).astype(np.float16).astype(np.float64)
        sc = np.clip(np.rint(a / np.where(d[:, :, None] == 0, 1, d[:, :, None])), 0, 15)
        mm = np.clip(np.rint(-b / np.where(dm[:, :, None] == 0, 1, dm[:, :, None])), 0, 15)
        blk[:, :, :16] = (sc.astype(np.uint8) | (mm.astype(np.uint8) << 4))
        blk[:, :, 80:82] = d.astype(np.float16)[:, :, None].view(np.uint8)
        blk[:, :, 82:84] = dm.astype(np.float16)[:, :, None].view(np.uint8)
        return out
    raise SystemExit(f"refit: type {qt.name} not supported")


def load(path):
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


def settings():
    """matching settings, all from the environment so every run's log names them"""
    return {"damp": float(os.environ.get("LG_MATCH_DAMP", "0.01")), "order": os.environ.get("LG_MATCH_ORDER") or None,
            "refit": os.environ.get("LG_MATCH_REFIT") == "1", "seq": os.environ.get("LG_SEQ_ACT") or None,
            "stock": os.environ.get("LG_MATCH_STOCK") == "1"}


def one(ref, don, act_dir, name):
    """match one matrix; returns new block bytes and a report (output error on fit rows and on held-out rows, stock
    rounding vs matched). LG_SEQ_ACT=<dir>: inputs recorded from the partly compressed model (same text, same rows as
    act_dir); the target becomes the ORIGINAL output given those drifted inputs, W' = W C Hq^-1 (C = X^T Xq / n), so
    this layer cancels upstream drift. LG_MATCH_REFIT=1: then re-fit the block scales to the chosen codes and re-match
    once on the new grid, kept only if the fit error is lower. LG_MATCH_ORDER=act: GPTQ act-order."""
    cfg = settings()
    rt, dt = ref[name], don[name]
    n_col = int(rt.shape[0])
    W = dequantize(rt.data, rt.tensor_type).reshape(-1, n_col).astype(np.float32)
    f = hessian(act_dir, name)
    if f is None:
        return None
    fq = hessian(cfg["seq"], name) if cfg["seq"] else None
    G = grams(f, n_col, fq)
    H, Hh = G["fit"], G["hold"]
    W0 = W
    if fq is not None:
        lam = cfg["damp"] * np.mean(np.diag(H)) + 1e-12
        Hd, Cd = H.copy(), G["C"].copy()
        Hd[np.diag_indices_from(Hd)] += lam; Cd[np.diag_indices_from(Cd)] += lam   # same ridge on both sides: no drift
        W = (W.astype(np.float64) @ Cd @ np.linalg.inv(Hd)).astype(np.float32)   # (X = Xq) gives back exactly W
        del Hd, Cd
    raw = np.asarray(dt.data).reshape(W.shape[0], -1)
    a, b, c0, qmax = grid(raw, dt.tensor_type, n_col)
    ref_vals = dequantize(dt.data, dt.tensor_type).reshape(W.shape)   # our reading of the format = the engine's decoder
    assert np.allclose(a * c0 + b, ref_vals, rtol=0, atol=1e-6 * np.abs(ref_vals).max()), f"{name}: grid decode mismatch"
    # LG_MATCH_STOCK=1: keep the donor's codes as they are and only measure (e.g. right-scale Q4_0, where there is
    # nothing left to match): the sidecar then holds the stock error on fit and held-out rows for the bit mix
    c1 = c0.copy() if cfg["stock"] else match(W, H, a, b, c0, qmax, cfg["damp"], order=cfg["order"])
    if cfg["refit"]:
        raw2 = refit(raw, dt.tensor_type, W, c1, np.diag(H)[None, :])
        a2, b2, _, _ = grid(raw2, dt.tensor_type, n_col)
        c2 = match(W, H, a2, b2, c1, qmax, cfg["damp"], order=cfg["order"])
        if out_err(a2 * c2 + b2 - W, H) < out_err(a * c1 + b - W, H):
            raw, a, b, c1 = raw2, a2, b2, c2
    new = pack(raw, dt.tensor_type, c1)
    got = dequantize(new, dt.tensor_type).reshape(W.shape)
    assert np.allclose(got, a * c1 + b, rtol=0, atol=1e-6 * np.abs(got).max()), f"{name}: packed codes/scales mismatch"
    E0, E1 = a * c0 + b - W, got - W                                    # stock rounding vs matched, same target
    rep = {"fit_stock": out_err(E0, H), "fit_matched": out_err(E1, H), "hold_stock": out_err(E0, Hh),
           "hold_matched": out_err(E1, Hh), "changed": float(np.mean(c0 != c1)), "n_fit": G["n_fit"], "n_hold": G["n_hold"]}
    if fq is not None:   # held-out rows: this layer on the drifted inputs vs the ORIGINAL layer on the original inputs
        def true_err(Q):
            Q = Q.astype(np.float64); W64 = W0.astype(np.float64)
            return float(np.einsum("ij,ij->", Q @ Hh, Q) - 2 * np.einsum("ij,ij->", W64 @ G["C_hold"], Q)
                         + np.einsum("ij,ij->", W64 @ G["X_hold"], W64))
        rep["hold_stock"], rep["hold_matched"] = true_err(a * c0 + b), true_err(got)
    return new, rep


def line(n, r):
    return (f"{n}: fit {r['fit_stock']:.4g} -> {r['fit_matched']:.4g} ({100 * (1 - r['fit_matched'] / r['fit_stock']):.1f}% less), "
            f"held-out {r['hold_stock']:.4g} -> {r['hold_matched']:.4g} ({100 * (1 - r['hold_matched'] / r['hold_stock']):.1f}% less), "
            f"codes changed {100 * r['changed']:.1f}%, rows {r['n_fit']}+{r['n_hold']}")


def build(ref_p, don_p, act_dir, out_p):
    ref, top = load(ref_p), gguf.GGUFReader(don_p)
    don = {t.name: t for t in top.tensors}
    h0 = hold_from(Path(act_dir) / "x")
    log(f"settings {settings()}, held out: " + (f"whole documents from row {h0}" if h0 is not None else f"every {HOLD}th row"))
    w = gguf.GGUFWriter(out_p, top.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=top.endianess)
    for f in top.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    data, side = {}, {}
    names = [t.name for t in top.tensors if re.match(r"blk\.\d+\.(%s)\.weight$" % "|".join(re.escape(a) for a in ACT), t.name)]
    keep = [n for n in names if don[n].tensor_type.name not in ("Q4_0", "Q3_K", "Q2_K")]
    if keep:   # e.g. stock Q4_0 gives the first layers' ffn_down Q4_1: those stay exactly as the donor has them
        log(f"kept as the donor has them (type not matched): " + ", ".join(f"{n} {don[n].tensor_type.name}" for n in keep))
    names = [n for n in names if n not in keep]
    cache = os.environ.get("LG_MATCH_CACHE")                       # shards share results through a cache folder:
    shard = os.environ.get("LG_MATCH_SHARD")                       # LG_MATCH_SHARD=i/n matches every n-th matrix only
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        cf = Path(cache) / (n + ".npz") if cache else None
        if cf and cf.exists():
            z = np.load(cf); r = (z["new"], json.loads(str(z["rep"])))
        elif shard and i % int(shard.split("/")[1]) != int(shard.split("/")[0]):
            continue
        else:
            r = one(ref, don, act_dir, n)
            if r is None:
                continue
            if cf:
                np.savez(cf, new=r[0], rep=json.dumps(r[1]))
            log(line(n, r[1]))
        if shard:
            continue
        data[n] = r[0]
        side[n] = {**r[1], "stock": r[1]["fit_stock"], "matched": r[1]["fit_matched"], "bytes": int(r[0].nbytes),
                   "type": don[n].tensor_type.name}
    if shard:
        log("shard done"); return
    for t in top.tensors:
        arr = data.get(t.name, t.data)
        w.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in top.tensors:
        w.write_tensor_data(data.get(t.name, t.data), tensor_endianess=top.endianess)
    w.close()
    Path(out_p + ".json").write_text(json.dumps({"settings": settings(), "matrices": side}, indent=1))
    tot = {k: sum(v[k] for v in side.values()) for k in ("fit_stock", "fit_matched", "hold_stock", "hold_matched")}
    log(f"built {out_p}: fit {tot['fit_stock']:.4g} -> {tot['fit_matched']:.4g}, held-out {tot['hold_stock']:.4g} -> {tot['hold_matched']:.4g}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        ref, don = load(sys.argv[2]), load(sys.argv[3])
        t0 = time.time()
        log(f"settings {settings()}")
        log(line(sys.argv[5], one(ref, don, sys.argv[4], sys.argv[5])[1]) + f", {time.time() - t0:.1f} s")
    elif cmd == "build":
        build(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
