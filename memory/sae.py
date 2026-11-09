"""Engram rung 1: find his fragments. A sparse autoencoder (SAE) on E4B's stream at a layer rewrites each word's
activity as a few active pieces out of many (TopK SAE, Gao et al. 2024: exactly k pieces on, dead pieces revived by
an auxiliary loss). Gemma Scope stops at Gemma 3, so E4B needs its own.

Public text only: FineWeb-Edu (web) and UltraChat 200k (chats, in Gemma's own chat format), half and half.

  python sae.py data  --model DIR --out DIR                 # tokenize rows of 512 into train.npy / eval.npy
  python sae.py train --model DIR --out DIR [--tokens N]    # one SAE per layer in LAYERS, all from the same pass
  python sae.py eval  --model DIR --out DIR                 # variance explained, dead pieces, loss when spliced in
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True   # the fragment finder's big matmuls: ~3x faster, plenty precise

LAYERS = [8, 12]           # the stream going INTO these layers (where rung 0 re-lights memories)
WIDTH = 32768              # pieces per layer
K = 64                     # pieces on per word
K_AUX = 512
DEAD_TOKENS = 10_000_000   # a piece that hasn't fired in this many words counts as dead
ROW = 512
ROWS_PER_SOURCE = 150_000
EVAL_ROWS = 128            # per source
BATCH = 4096
HALF = 2 ** 17             # activation buffer = 2 halves, shuffled together
FWD_ROWS = 32
LR = 1.5e-4


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n=WIDTH, k=K):
        super().__init__()
        self.k = k
        dec = torch.randn(n, d)
        dec /= dec.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(dec)
        self.W_enc = torch.nn.Parameter(dec.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.ones(()))   # inputs are divided by this (mean squared norm -> d)

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x, k=None):
        v, i = self.pre(x).topk(k or self.k, dim=-1)
        return F.relu(v), i

    def decode(self, v, i):
        z = torch.zeros(*v.shape[:-1], self.W_dec.shape[0], device=v.device, dtype=v.dtype).scatter(-1, i, v)
        return z @ self.W_dec + self.b_dec

    def roundtrip(self, h, k=None):
        """his stream -> pieces -> his stream (any shape [..., d], any dtype)"""
        x = h.float() / self.scale
        return (self.decode(*self.encode(x, k)) * self.scale).to(h.dtype)


def load_sae(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    sae = TopKSAE(ck["W_dec"].shape[1], ck["W_dec"].shape[0], ck["k"]).to(device)
    sae.load_state_dict(ck["state"])
    return sae.eval()


def load_model(model_dir, attn="sdpa"):
    """newer transformers refuse sdpa for Gemma 4's audio part (2026-11-08), so the text part gets `attn` and the
    audio and vision parts, which the finder never runs, get eager"""
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation={"text_config": attn, "vision_config": "eager", "audio_config": "eager"}).eval()
    except (ValueError, TypeError):
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda").eval()
    print("text attention:", model.config.text_config._attn_implementation, flush=True)
    return tok, model


# ---------------------------------------------------------------- data
def cmd_data(a):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def rows_from(texts, need, special):
        got = []
        for s in range(0, len(texts), 2000):
            enc = tok([t[:4000] for t in texts[s:s + 2000]], add_special_tokens=special)["input_ids"]
            got += [e[:ROW] for e in enc if len(e) >= ROW]
            if len(got) >= need:
                break
        return got[:need]

    web_rows = []
    pf = pq.ParquetFile(hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet",
                                        repo_type="dataset"))
    for b in pf.iter_batches(batch_size=20000, columns=["text"]):
        web_rows += rows_from(b.column("text").to_pylist(), ROWS_PER_SOURCE + EVAL_ROWS - len(web_rows), True)
        if len(web_rows) >= ROWS_PER_SOURCE + EVAL_ROWS:
            break
    print("web rows", len(web_rows), flush=True)

    def chats(fname):
        msgs = pq.read_table(hf_hub_download("HuggingFaceH4/ultrachat_200k", fname, repo_type="dataset"),
                             columns=["messages"]).column("messages").to_pylist()
        return [tok.apply_chat_template(m, tokenize=False) for m in msgs]
    names = ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
             "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
             "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"]
    chat_rows = []
    for n in names:
        chat_rows += rows_from(chats(n), ROWS_PER_SOURCE - len(chat_rows), False)
        if len(chat_rows) >= ROWS_PER_SOURCE:
            break
    chat_eval = rows_from(chats("data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"), EVAL_ROWS, False)
    print("chat rows", len(chat_rows), "chat eval", len(chat_eval), flush=True)

    train = np.array(web_rows[EVAL_ROWS:] + chat_rows, dtype=np.int32)
    np.random.default_rng(2026).shuffle(train)
    np.save(out / "train.npy", train)
    np.save(out / "eval.npy", np.array(web_rows[:EVAL_ROWS] + chat_eval, dtype=np.int32))
    json.dump(dict(train_rows=len(train), web=len(web_rows) - EVAL_ROWS, chat=len(chat_rows), row=ROW),
              open(out / "data.json", "w"))
    print("train", train.shape, flush=True)


# ---------------------------------------------------------------- his stream at the layers
class Stop(Exception):
    pass


class Tap:
    """Run his text layers only up to the last tapped layer; return the stream going into each tapped layer."""

    def __init__(self, model, layers):
        self.lm = model.model.language_model
        self.layers, self.got = layers, None
        for L in layers:
            self.lm.layers[L].register_forward_pre_hook(self._hook(L), with_kwargs=True)

    def _hook(self, L):
        def hook(mod, args, kwargs):
            if self.got is None:
                return None
            self.got[L] = (args[0] if args else kwargs["hidden_states"])[:, 1:]   # every word but <bos>
            if L == max(self.layers):
                raise Stop
        return hook

    @torch.no_grad()
    def __call__(self, ids):
        self.got = {}
        try:
            self.lm(input_ids=ids)
        except Stop:
            pass
        got, self.got = self.got, None
        return {L: t.reshape(-1, t.shape[-1]) for L, t in got.items()}


# ---------------------------------------------------------------- train
def cmd_train(a):
    out = Path(a.out)
    tok, model = load_model(a.model)
    tap = Tap(model, LAYERS)
    rows = np.load(out / "train.npy", mmap_mode="r")
    cursor, spare = [0], {L: [] for L in LAYERS}    # spare = words read past the last half, used first next time

    def fresh(n):
        """the next n words' activations, or None once the data runs out (nothing read is thrown away)"""
        acts = {L: list(spare[L]) for L in LAYERS}
        have = sum(t.shape[0] for t in acts[LAYERS[0]])
        while have < n:
            if cursor[0] >= len(rows):
                return None
            r = torch.from_numpy(np.array(rows[cursor[0]:cursor[0] + FWD_ROWS])).cuda()
            cursor[0] += FWD_ROWS
            for L, t in tap(r).items():
                acts[L].append(t)
            have += r.shape[0] * (ROW - 1)
        got = {L: torch.cat(v) for L, v in acts.items()}
        for L in LAYERS:
            spare[L] = [got[L][n:]]
        return {L: t[:n] for L, t in got.items()}

    def save(seen):
        for L in LAYERS:
            torch.save(dict(state=saes[L].state_dict(), W_dec=saes[L].W_dec.detach(), k=K, layer=L,
                            words=seen), out / f"sae_L{L}.pt")

    d = model.config.text_config.hidden_size
    keep = fresh(HALF)
    saes, opts, last_fired = {}, {}, {}
    for L in LAYERS:
        x = keep[L].float()
        sae = TopKSAE(d).cuda()
        sae.scale.fill_(x.pow(2).sum(-1).mean().div(d).sqrt().item())
        sae.b_dec.data.copy_((x / sae.scale).mean(0))
        saes[L] = sae
        opts[L] = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999), eps=6.25e-10)
        last_fired[L] = torch.zeros(WIDTH, dtype=torch.long, device="cuda")
    # the cool-down has to end inside the data: the last kept half is never trained on
    total_steps = min(a.tokens, len(rows) * (ROW - 1) - HALF) // BATCH
    step, seen, t0 = 0, 0, time.time()
    log = open(out / "train.log", "a")
    stats = {L: dict(fve=0.0, aux=0.0, n=0) for L in LAYERS}
    while step < total_steps:
        new = fresh(HALF)
        if new is None:
            break
        perm = torch.randperm(2 * HALF, device="cuda")
        bufs = {}
        for L in LAYERS:
            bufs[L] = torch.cat([keep[L], new[L]])[perm]
            keep[L] = bufs[L][HALF:].clone()
        del new
        for s in range(0, HALF, BATCH):
            lr = LR * min(1.0, (step + 1) / 1000) * min(1.0, (total_steps - step) / (0.2 * total_steps))
            for L in LAYERS:
                sae, opt = saes[L], opts[L]
                for g in opt.param_groups:
                    g["lr"] = lr
                x = bufs[L][s:s + BATCH].float() / sae.scale
                pre = sae.pre(x)
                v, i = pre.topk(K, dim=-1)
                v = F.relu(v)
                xh = sae.decode(v, i)
                var = (x - x.mean(0)).pow(2).sum(-1).mean()
                mse = (xh - x).pow(2).sum(-1).mean() / var
                fired = torch.zeros(WIDTH, dtype=torch.bool, device="cuda")
                fired[i[v > 0]] = True
                last_fired[L][fired] = seen + BATCH
                dead = (seen - last_fired[L]) > DEAD_TOKENS
                aux = torch.zeros((), device="cuda")
                if dead.any():
                    dp = pre.masked_fill(~dead, -1e9)
                    av, ai = dp.topk(min(K_AUX, int(dead.sum())), dim=-1)
                    err = (x - xh).detach()
                    eh = torch.zeros_like(pre).scatter(-1, ai, F.relu(av)) @ sae.W_dec
                    aux = (eh - err).pow(2).sum(-1).mean() / (err - err.mean(0)).pow(2).sum(-1).mean()
                    aux = torch.nan_to_num(aux)
                opt.zero_grad(set_to_none=True)
                (mse + aux / 32).backward()
                with torch.no_grad():   # keep decoder rows unit length: drop the gradient along each row
                    Wd = sae.W_dec
                    Wd.grad -= (Wd.grad * Wd).sum(-1, keepdim=True) * Wd
                opt.step()
                with torch.no_grad():
                    sae.W_dec /= sae.W_dec.norm(dim=-1, keepdim=True)
                st = stats[L]
                st["fve"] += 1 - mse.item()
                st["aux"] += aux.item()
                st["n"] += 1
            step += 1
            seen += BATCH
        del bufs
        if seen // 5_000_000 == (seen - HALF) // 5_000_000 and step < total_steps:
            continue
        msg = f"{seen / 1e6:7.1f}M words  {seen / (time.time() - t0):7.0f}/s  " + "  ".join(
            f"L{L} explained {stats[L]['fve'] / max(1, stats[L]['n']):.3f} "
            f"dead {(((seen - last_fired[L]) > DEAD_TOKENS).float().mean().item()):.3f}" for L in LAYERS)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        stats = {L: dict(fve=0.0, aux=0.0, n=0) for L in LAYERS}
        if seen // 20_000_000 != (seen - HALF) // 20_000_000:
            save(seen)
    save(seen)
    msg = f"done at {seen / 1e6:.1f}M words ({step} of {total_steps} steps)"
    print(msg, flush=True)
    log.write(msg + "\n")


# ---------------------------------------------------------------- eval
def cmd_eval(a):
    out = Path(a.out)
    tok, model = load_model(a.model)
    lm = model.model.language_model
    rows = torch.from_numpy(np.load(out / "eval.npy").astype(np.int64)).cuda()
    res = {}
    for L in LAYERS:
        sae = load_sae(out / f"sae_L{L}.pt")
        swap = {"on": False}

        def hook(mod, args, kwargs, sae=sae, swap=swap):
            if not swap["on"]:
                return None
            hs = args[0].clone()
            hs[:, 1:] = sae.roundtrip(hs[:, 1:])
            return (hs,) + tuple(args[1:]), kwargs
        hd = lm.layers[L].register_forward_pre_hook(hook, with_kwargs=True)
        clean = spliced = 0.0
        ex = var = 0.0
        fired = torch.zeros(WIDTH, dtype=torch.bool, device="cuda")
        cap = {}
        hc = lm.layers[L].register_forward_pre_hook(lambda m, a_, k_: cap.__setitem__("h", a_[0][:, 1:]),
                                                    with_kwargs=True)
        with torch.no_grad():
            for s in range(0, len(rows), 4):
                r = rows[s:s + 4]
                for on in (False, True):
                    swap["on"] = on
                    lg = model(input_ids=r).logits.float()
                    ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), r[:, 1:].reshape(-1)).item()
                    if on:
                        spliced += ce
                    else:
                        clean += ce
                        x = cap["h"].reshape(-1, cap["h"].shape[-1]).float() / sae.scale
                        v, i = sae.encode(x)
                        xh = sae.decode(v, i)
                        ex += (xh - x).pow(2).sum().item()
                        var += (x - x.mean(0)).pow(2).sum().item()
                        fired[i[v > 0]] = True
        hd.remove()
        hc.remove()
        n = math.ceil(len(rows) / 4)
        res[L] = dict(explained=1 - ex / var, loss_clean=clean / n, loss_spliced=spliced / n,
                      loss_added=(spliced - clean) / n, never_fired=1 - fired.float().mean().item(),
                      words_trained=torch.load(out / f"sae_L{L}.pt", map_location="cpu")["words"])
        print(L, res[L], flush=True)
    json.dump(res, open(out / "eval.json", "w"), indent=1)


def main():
    global LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "eval"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=150_000_000)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)),
                    help="the floors to find pieces at, all from the same pass (2026-11-08: 5,6 for bricks)")
    a = ap.parse_args()
    LAYERS = [int(x) for x in a.layers.split(",")]
    dict(data=cmd_data, train=cmd_train, eval=cmd_eval)[a.cmd](a)


if __name__ == "__main__":
    main()
