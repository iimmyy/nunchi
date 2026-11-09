"""Engram rung 0: can his frozen brain read a moment back from its own saved states?

A made-up moment is read once in one chat (context A) and its states are saved. Later, in a different chat
(context B), a question about it is asked with the moment put back right before the question, in one of these ways:

  text        the moment's words, read fresh (ceiling)
  none        nothing (floor: what guessing gets)
  kv          the saved keys and values of layers 0-23 (layers 24-41 reuse 22/23), re-aimed to the new positions:
              short-term memory = his own KV cache, kept
  kvsal       same, only the most surprising word pieces of the moment
  kvsalw      same budget, but whole words: a word is kept or dropped as one (its most surprising piece decides)
  kvswap      the saved KV of a DIFFERENT moment of the same kind (cheat check: does the answer follow the memory?)
  resL        the moment's layer-L stream re-lit at layer L; below L the moment does not exist (masked);
              per-layer inputs at those positions are the pad token's, as for sound (long-term path, exact states)
  pleL        resL plus the moment's own per-layer inputs at layers >= L (they carry the word identity: an upper
              bound, not something long-term memory would have)
  salL        resL with only the most surprising words kept
  swapL       resL with a different moment's states (cheat check)
  saeL        resL after a trip through his fragments (rung 1: stream -> SAE pieces -> stream)
  saepleL     saeL plus the word identity (the "name tag", Immy's option A)
  saepleLkN   saepleL keeping only the N strongest pieces of 64 (32 at both layers; 16 and 8 at layer 8: fading)
  saepleswap8 saeple8 with a different moment's pieces (cheat check for the bricks)
  WAY_after   the same memory put right AFTER the question instead of before it (the order the engine would use:
              hear the message, be reminded, answer; nothing has to be re-read)
  WAY_recall  after the question, at the start of his own turn, with its source: You told me: "<memory>"
              (Immy's bricks test: re-lit pieces are rebuilt there from layer L up, listening to the question and
              his own turn; a saved KV memory was made in another chat and can't)

Scores: the greedy answer (does it contain the right value; did he refuse) and the rank of the right value among the
kind's 10 possible values (log-probability of each as the answer). Made-up moments only, no real chats.

  python rung0.py --model DIR --out DIR [--check] [--sae 8:sae_L8.pt,12:sae_L12.pt] [--ways text,none,...]
                  [--layouts before,after,recall]
"""
import argparse, json, math, re, time
from pathlib import Path

import torch

from moments import KINDS, PER_KIND, REFUSED, SMALL_TALK, moments, prefix

LS = [4, 8, 12, 16, 20, 22, 28]        # 28 is past the KV-sharing line on purpose (should look like nothing)
SAL_LS = [8, 12, 16, 20, 22]
N_KV = 24                               # layers 0-23 make their own keys and values
SAL_FRAC = 0.4
LATER = {"text", "none", "kv", "kvsalw", "kvswap", "res4", "ple4", "res8", "ple8", "sae8", "saeple8", "saeple8k32",
         "saeple8k16", "saeple8k8", "saepleswap8", "res12", "ple12", "sae12", "saeple12"}   # also run after / recall
SAE_KS = {8: (None, 32, 16, 8)}         # pieces kept per word (None = all 64); other layers: (None, 32)
MAX_NEW = 12


class Splicer:
    """Hooks on his text layers. cfg=None: untouched. Positions [s, s+n) are the memory slot."""

    def __init__(self, layers):
        self.cfg, self.cap = None, None
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(self._layer_hook(i), with_kwargs=True)
            if i < N_KV:
                layer.self_attn.k_norm.register_forward_hook(self._kv_hook(i, "k"))
                layer.self_attn.v_norm.register_forward_hook(self._kv_hook(i, "v"))

    def _kv_hook(self, i, which):
        def hook(mod, inp, out):
            c = self.cfg
            if c is None:
                return None
            s, n = c["s"], c["n"]
            if c.get("capture"):
                self.cap[which][i] = out[:, s:s + n].detach().clone()
                return None
            if i in c.get("kv_layers", ()):
                out = out.clone()
                out[:, s:s + n] = c["mem"][which][i].to(out.dtype)
                return out
            return None
        return hook

    def _layer_hook(self, i):
        def hook(mod, args, kwargs):
            c = self.cfg
            if c is None:
                return None
            s, n = c["s"], c["n"]
            hs = args[0] if args else kwargs["hidden_states"]
            ple = args[1] if len(args) > 1 else kwargs.get("per_layer_input")
            if c.get("capture"):
                self.cap["h"][i] = hs[:, s:s + n].detach().clone()
                self.cap["ple"][i] = ple[:, s:s + n].detach().clone()
                return None
            L = c.get("L")
            if L is None:
                return None
            if i == L:
                hs = hs.clone()
                hs[:, s:s + n] = c["mem"]["h"][i].to(hs.dtype)
            if c.get("ple") and i >= L:
                ple = ple.clone()
                ple[:, s:s + n] = c["mem"]["ple"][i].to(ple.dtype)
            if c.get("block") and i < L:
                mask = kwargs["attention_mask"].clone()
                if mask.dtype == torch.bool:
                    mask[..., s:s + n] = False
                else:
                    mask[..., s:s + n] = torch.finfo(mask.dtype).min
                kwargs["attention_mask"] = mask
            return (hs, ple) + tuple(args[2:]), kwargs
        return hook


def subset(mem, idx):
    return {k: {i: t[:, idx] for i, t in d.items()} for k, d in mem.items()}


class Rig:
    def __init__(self, model_dir, dtype=torch.bfloat16):
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="eager", device_map="cuda").eval()
        self.layers = self.model.model.language_model.layers
        self.sp = Splicer(self.layers)
        self.pad = self.tok.pad_token_id
        self.stop = {self.tok.convert_tokens_to_ids("<turn|>"), self.tok.eos_token_id}

    def ids(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def logits(self, rows, cfg=None, **kw):
        width = max(len(r) for r in rows)
        x = torch.tensor([r + [self.pad] * (width - len(r)) for r in rows], device="cuda")
        self.sp.cfg = cfg
        try:
            return self.model(input_ids=x, **kw).logits.float()
        finally:
            self.sp.cfg = None

    def capture(self, pre, span):
        """Read pre+span (+ end of turn) once, save the span's states."""
        rows = [pre + span + self.ids("<turn|>\n")]
        self.sp.cap = {"k": {}, "v": {}, "h": {}, "ple": {}}
        lg = self.logits(rows, {"capture": True, "s": len(pre), "n": len(span)})
        mem, self.sp.cap = self.sp.cap, None
        lp = torch.log_softmax(lg[0], -1)
        surprise = [-lp[len(pre) + j - 1, t].item() for j, t in enumerate(span)]
        return mem, surprise

    def answer(self, row, cfg):
        out = []
        for _ in range(MAX_NEW):
            t = int(self.logits([row + out], cfg)[0, -1].argmax())
            if t in self.stop:
                break
            out.append(t)
        return self.tok.decode(out, skip_special_tokens=True).strip()

    def cand_scores(self, row, cfg, cands):
        variants = [(c, v) for c in cands for v in dict.fromkeys([c, c[:1].upper() + c[1:]])]
        rows = [row + self.ids(v) for _, v in variants]
        lp = torch.log_softmax(self.logits(rows, cfg), -1)
        score = {c: [] for c in cands}
        for r, (c, v) in enumerate(variants):
            toks = rows[r][len(row):]
            score[c].append(sum(lp[r, len(row) + j - 1, t].item() for j, t in enumerate(toks)))
        return {c: torch.logsumexp(torch.tensor(v), 0).item() for c, v in score.items()}


def has(text, value):
    return re.search(r"\b" + re.escape(value.lower()) + r"\b", text.lower()) is not None


def whole_words(tok, span, surprise, frac):
    """the most surprising whole words, as positions, until they hold frac of the pieces (same budget as kvsal)"""
    words = []
    for j, piece in enumerate(tok.convert_ids_to_tokens(span)):
        if not words or piece.startswith("\u2581"):
            words.append([])
        words[-1].append(j)
    words.sort(key=lambda w: -max(surprise[j] for j in w))
    keep, need = [], max(3, math.ceil(frac * len(span)))
    for w in words:
        if len(keep) >= need:
            break
        keep += w
    return sorted(keep)


def conditions(m, mem, swap_mem, n, sal, n_swap, sae_mems=None, salw=None, swap_sae=None):
    """(name, slot length, cfg builder for slot start s, which answer counts)"""
    out = [("text", None, None, "answer"), ("none", 0, None, "answer")]
    for (L, k), sm_ in (sae_mems or {}).items():
        tag = f"{L}" if k is None else f"{L}k{k}"
        if k is None:
            out.append((f"sae{tag}", n, lambda s, L=L, sm_=sm_: dict(s=s, n=n, mem=sm_, L=L, block=True), "answer"))
        out.append((f"saeple{tag}", n,
                    lambda s, L=L, sm_=sm_: dict(s=s, n=n, mem=sm_, L=L, block=True, ple=True), "answer"))
    if swap_sae and (8, None) in swap_sae:
        ssm = swap_sae[(8, None)]
        out.append(("saepleswap8", n_swap,
                    lambda s: dict(s=s, n=n_swap, mem=ssm, L=8, block=True, ple=True), "swap"))
    kv = list(range(N_KV))
    out.append(("kv", n, lambda s: dict(s=s, n=n, mem=mem, kv_layers=kv), "answer"))
    sm = subset(mem, sal)
    out.append(("kvsal", len(sal), lambda s: dict(s=s, n=len(sal), mem=sm, kv_layers=kv), "answer"))
    if salw is not None:
        smw = subset(mem, salw)
        out.append(("kvsalw", len(salw), lambda s: dict(s=s, n=len(salw), mem=smw, kv_layers=kv), "answer"))
    out.append(("kvswap", n_swap, lambda s: dict(s=s, n=n_swap, mem=swap_mem, kv_layers=kv), "swap"))
    for L in LS:
        out.append((f"res{L}", n, lambda s, L=L: dict(s=s, n=n, mem=mem, L=L, block=True), "answer"))
        out.append((f"ple{L}", n, lambda s, L=L: dict(s=s, n=n, mem=mem, L=L, block=True, ple=True), "answer"))
        out.append((f"swap{L}", n_swap,
                    lambda s, L=L: dict(s=s, n=n_swap, mem=swap_mem, L=L, block=True), "swap"))
    for L in SAL_LS:
        out.append((f"sal{L}", len(sal), lambda s, L=L: dict(s=s, n=len(sal), mem=sm, L=L, block=True), "answer"))
    return out


def check(rig, ms):
    """The splice must be exact where it can be: same chat, same positions -> same answer logits as the text."""
    ok = True
    s_tpl = rig.tok.apply_chat_template(
        [{"role": "user", "content": SMALL_TALK[0][0]}, {"role": "assistant", "content": SMALL_TALK[0][1]},
         {"role": "user", "content": "Hello."}], tokenize=False, add_generation_prompt=True)
    mine = prefix([0]) + "Hello.<turn|>\n<|turn>model\n"
    print("chat format same as the template:", s_tpl == mine)
    ok &= s_tpl == mine
    for m in ms[:3]:
        pre, span = rig.ids(prefix(m["ctx_a"])), rig.ids(m["text"])
        suf = rig.ids("\n\n" + m["question"] + "<turn|>\n<|turn>model\n")
        mem, _ = rig.capture(pre, span)
        s, n = len(pre), len(span)
        ref = rig.logits([pre + span + suf])[0, -1]
        # rounding floor: the same text in a batch of a different shape (the GPU picks other kernels)
        noise = (rig.logits([pre + span + suf, pre + suf])[0, len(pre + span + suf) - 1] - ref).abs().max().item()
        slot = pre + [rig.pad] * n + suf
        kv = rig.logits([slot], dict(s=s, n=n, mem=mem, kv_layers=list(range(N_KV))))[0, -1]
        full = rig.logits([slot], dict(s=s, n=n, mem=mem, L=12, ple=True, kv_layers=list(range(12))))[0, -1]
        d_kv, d_full = (kv - ref).abs().max().item(), (full - ref).abs().max().item()
        same = int(kv.argmax()) == int(ref.argmax()) == int(full.argmax())
        print(f"{m['kind']}: kv splice vs text max diff {d_kv:.4f}; layer-12 splice + KV below vs text {d_full:.4f};"
              f" same text, other batch shape {noise:.4f}; same top word {same}")
        ok &= max(d_kv, d_full) <= max(0.02, 3 * noise) and same
        rig.sp.cfg = dict(s=s, n=n, mem=mem, L=12, block=True)
        try:
            with torch.no_grad():
                att = rig.model(input_ids=torch.tensor([slot], device="cuda"), output_attentions=True).attentions
        finally:
            rig.sp.cfg = None
        below = max(att[i][0, :, s + n:, s:s + n].abs().max().item() for i in range(12))
        above = att[12][0, :, s + n:, s:s + n].abs().max().item()
        print(f"   question words looking at the slot: below layer 12 max {below:.2e}, at layer 12 max {above:.3f}")
        ok &= below == 0.0 and above > 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="full precision (for the exactness check)")
    ap.add_argument("--sae", default="", help="L:path,... trained fragment finders (sae.py)")
    ap.add_argument("--ways", default="", help="only these ways (comma list, e.g. kv,saeple8_recall); default all")
    ap.add_argument("--layouts", default="before,after,recall", help="which orders to run (comma list)")
    a = ap.parse_args()
    orders = a.layouts.split(",")
    assert set(orders) <= {"before", "after", "recall"}, orders
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rig = Rig(a.model, torch.float32 if a.fp32 else torch.bfloat16)
    ms = moments()
    if a.check:
        good = check(rig, ms)
        print("CHECK", "PASSED" if good else "FAILED")
        raise SystemExit(0 if good else 1)

    saved = []
    for m in ms:                        # read every moment once, in its own chat (context A)
        pre, span = rig.ids(prefix(m["ctx_a"])), rig.ids(m["text"])
        mem, surprise = rig.capture(pre, span)
        k = max(3, math.ceil(SAL_FRAC * len(span)))
        sal = sorted(sorted(range(len(span)), key=lambda j: -surprise[j])[:k])
        saved.append(dict(mem=mem, span=span, sal=sal, salw=whole_words(rig.tok, span, surprise, SAL_FRAC),
                          surprise=surprise))
    one = saved[0]
    kv_bytes = sum(t[0, 0].numel() * 2 for d in ("k", "v") for t in one["mem"][d].values())
    saes = {}
    for item in filter(None, a.sae.split(",")):
        from sae import load_sae
        L, path = item.split(":", 1)
        saes[int(L)] = load_sae(path)
    for L, sae in saes.items():
        hs = torch.cat([sv["mem"]["h"][L][0] for sv in saved]).float()
        with torch.no_grad():
            rt = sae.roundtrip(hs)
        fve = 1 - (rt - hs).pow(2).sum() / (hs - hs.mean(0)).pow(2).sum()
        print(f"layer {L}: fragments explain {fve.item():.3f} of the memory words' stream", flush=True)
        for sv in saved:
            with torch.no_grad():
                for k in SAE_KS.get(L, (None, 32)):
                    sv.setdefault("sae", {})[(L, k)] = {**sv["mem"], "h": {**sv["mem"]["h"],
                                                                         L: sae.roundtrip(sv["mem"]["h"][L], k)}}
    ways = set(filter(None, a.ways.split(",")))
    print(f"saved per word: KV of layers 0-{N_KV - 1} = {kv_bytes / 1024:.1f} KB (bf16), "
          f"one layer's stream = {one['mem']['h'][12][0, 0].numel() * 2 / 1024:.1f} KB", flush=True)

    rows_out = open(out / "rung0.jsonl", "w")
    t0 = time.time()
    for i, m in enumerate(ms):
        sv, sw = saved[i], saved[m["swap"]]
        n, n_swap = len(sv["span"]), len(sw["span"])
        pre = rig.ids(prefix(m["ctx_b"]))
        # (chat B up to the memory, chat B after it) for each order
        around = {"before": ([], rig.ids("\n\n" + m["question"] + "<turn|>\n<|turn>model\n")),
                  "after": (rig.ids(m["question"] + "\n\n"), rig.ids("<turn|>\n<|turn>model\n")),
                  "recall": (rig.ids(m["question"] + "<turn|>\n<|turn>model\n" + 'You told me: "'), rig.ids('"\n\n'))}
        layouts = []
        for name, slot_n, build, target in conditions(m, sv["mem"], sw["mem"], n, sv["sal"], n_swap,
                                                      sv.get("sae"), sv["salw"], sw.get("sae")):
            for order in orders:
                if order == "before" or name in LATER:
                    layouts.append((name if order == "before" else f"{name}_{order}", slot_n, build, target, order))
        for name, slot_n, build, target, order in layouts:
            if ways and name not in ways:
                continue
            slot = sv["span"] if name.startswith("text") else [rig.pad] * slot_n
            lead, tail = around[order]
            row = pre + lead + slot + tail
            s = len(pre) + len(lead)
            cfg = build(s) if build else None
            ans = rig.answer(row, cfg)
            sc = rig.cand_scores(row, cfg, m["cands"])
            want = m["answer"] if target == "answer" else ms[m["swap"]]["answer"]
            best = max(sc, key=sc.get)
            p = torch.softmax(torch.tensor(list(sc.values())), 0)[list(sc).index(want)].item()
            rec = dict(i=i, kind=m["kind"], cond=name, order=order, moment=m["text"], question=m["question"],
                       want=want, own=m["answer"], said=ans, said_right=has(ans, want), said_own=has(ans, m["answer"]),
                       refused=not has(ans, want) and REFUSED.search(ans) is not None,
                       top=best, top_right=best == want, p_right=p, slot=slot_n,
                       distractor_top=(best == m["distractor"]) if m["distractor"] else None,
                       kept=[rig.tok.decode([sv["span"][j]]) for j in sv["salw" if "salw" in name else "sal"]]
                       if "sal" in name else None)
            rows_out.write(json.dumps(rec) + "\n")
        rows_out.flush()
        print(f"{i + 1}/{len(ms)} moments done, {time.time() - t0:.0f} s", flush=True)
    rows_out.close()

    recs = [json.loads(l) for l in open(out / "rung0.jsonl")]
    names = list(dict.fromkeys(r["cond"] for r in recs))
    summary = {}
    print(f"\n{'way':18} {'said right':>10} {'refused':>8} {'ranked 1st':>10} {'p(right)':>9}"
          "   (10 choices each, chance 10%)")
    for c in names:
        rs = [r for r in recs if r["cond"] == c]
        sr = sum(r["said_right"] for r in rs) / len(rs)
        rf = sum(r["refused"] for r in rs) / len(rs)
        tr = sum(r["top_right"] for r in rs) / len(rs)
        pr = sum(r["p_right"] for r in rs) / len(rs)
        summary[c] = dict(n=len(rs), said_right=sr, refused=rf, ranked_first=tr, p_right=pr,
                          by_kind={k: sum(r["top_right"] for r in rs if r["kind"] == k) / PER_KIND for k in KINDS})
        print(f"{c:18} {sr:10.1%} {rf:8.1%} {tr:10.1%} {pr:9.3f}")
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
