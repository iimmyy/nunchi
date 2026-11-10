"""Hand out bits by READING TIME, not bytes (2026-11-10).

The read-speed test (kernel2/kq*.txt) showed that on the Haswell stand-in a table's time is not its size: imi's 4-bit
reader is at the edge of both memory and math, so a 2-bit table (58% of the bytes) reads in ~80% of the time with a
plain 8-row reader. So the budget here is time: cost(type) per number, relative to imi's 4-bit reader.

Units: the engine reads some tables together (core/seoul.cpp): attention q, k, v and gate as one table (qkvg), the
feed-forward gate and up as one (gate_up). Those share one type. Others (in_proj, out_proj, attn_output, ffn_down)
each get their own.

Harm of type t for table T (predicted whole-brain drift from the 16-bit brain, KL on Korean text):
    harm_T(t) = dKL_T * (r_T(t) - r_T(base)) / (r_T(Q2_K) - 1)
dKL_T: the whole-brain test with only T at plain 2 bits (sens/sens.tsv, against the box-made all-4-bit brain).
r_T(t): T's output error at type t over its plain 4-bit error, averaged over the three jobs (hearing, translation,
voice) so no job is left out (pieces-jobs/jobs.tsv; pieces-hi for 8 bits). Attention gates were not tested alone:
they take their layer's median factor.

  python3 alloc2.py JOBS.tsv BASE TYPES COSTS [budget ...]   e.g. pieces-jobs/jobs.tsv room room,mQ2_K,q8 1,0.8,1.7
Writes ladder/t<budget>-<types>.choice (tensor<TAB>label, base tables left out) and prints the predicted harm."""
import collections, csv, os, sys
import numpy as np

BOX_Q4 = 0.101398
LABEL = {"mQ2_K": "mq2", "mQ3_K": "mq3", "m3Q2_K": "m3q2", "m3Q3_K": "m3q3", "m3Q4_0": "m3q4", "q8": "q8", "Q2_K": "q2", "Q3_K": "q3"}


def load(jobs_p):
    J = collections.defaultdict(dict)
    for r in csv.DictReader(open(jobs_p), delimiter="\t"):
        J[r["table"]][r["version"]] = [float(r[k]) for k in ("hear", "tr", "voice")]
    hi = {r["table"]: r for r in csv.DictReader(open("pieces-hi/tables.tsv"), delimiter="\t")}
    tab = {r["table"]: r for r in csv.DictReader(open("pieces/tables.tsv"), delimiter="\t")}
    sens = {r[0]: float(r[1]) for r in csv.reader(open("sens/sens.tsv"), delimiter="\t")}
    T = {}
    for name in J:
        q4 = J[name]["Q4_K"]
        r = {v: float(np.mean([e / b for e, b in zip(J[name][v], q4)])) for v in J[name]}
        r["q8"] = float(hi[name]["Q8_0_true"]) / float(tab[name]["Q4_K_true"])
        d = sens.get(name + ":q2_K")
        T[name] = {"n": int(tab[name]["rows"]) * int(tab[name]["cols"]), "r": r, "dkl": None if d is None else max(d - BOX_Q4, 0.0)}
    for name, t in T.items():   # attention gates: their layer's median factor (dKL per unit of r)
        if t["dkl"] is None:
            L = name.split(".")[1]
            fs = [u["dkl"] / (u["r"]["Q2_K"] - 1) for m, u in T.items() if m.split(".")[1] == L and ".attn_" in m and u["dkl"] is not None]
            t["dkl"] = float(np.median(fs)) * (t["r"]["Q2_K"] - 1)
    return T


def units(T):
    U = collections.defaultdict(list)
    for name in T:
        L, kind = name.split(".")[1], name.split(".", 2)[2].replace(".weight", "")
        key = {"attn_q": "qkvg", "attn_k": "qkvg", "attn_v": "qkvg", "attn_gate": "qkvg", "ffn_gate": "gate_up", "ffn_up": "gate_up"}.get(kind, kind)
        U[f"blk.{L}.{key}"].append(name)
    return dict(U)


def harm(t, ty, base):
    return t["dkl"] * (t["r"][ty] - t["r"][base]) / (t["r"]["Q2_K"] - 1)


def solve(T, U, base, types, cost, budget):
    """minimize total harm with total time <= budget (time relative to every table in imi's 4-bit reader = 1)"""
    N = sum(t["n"] for t in T.values())
    opts = {u: [(sum(harm(T[n], ty, base) for n in ns), sum(T[n]["n"] for n in ns) * cost[ty] / N, ty) for ty in types] for u, ns in U.items()}
    def pick(lam):
        ch = {u: min(o, key=lambda x: x[0] + lam * x[1]) for u, o in opts.items()}
        return ch, sum(c[1] for c in ch.values()), sum(c[0] for c in ch.values())
    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget: hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[1] > budget: lo = mid
        else: hi = mid
    return pick(hi)


if __name__ == "__main__":
    jobs_p, base, types, costs = sys.argv[1:5]
    types = types.split(","); cost = dict(zip(types, map(float, costs.split(","))))
    budgets = [float(b) for b in sys.argv[5:]] or [1.0, 0.95, 0.9, 0.85, 0.8]
    T = load(jobs_p); U = units(T)
    N = sum(t["n"] for t in T.values())
    print(f"base {base}; types {types}; time cost per number {cost}; {len(T)} tables in {len(U)} units")
    os.makedirs("ladder", exist_ok=True)
    for b in budgets:
        ch, tm, h = solve(T, U, base, types, cost, b)
        share = collections.Counter()
        for u, c in ch.items(): share[c[2]] += sum(T[n]["n"] for n in U[u]) / N
        bits = {"room": 4.5, "m3Q4_0": 4.5, "mQ2_K": 2.625, "m3Q2_K": 2.625, "mQ3_K": 3.4375, "m3Q3_K": 3.4375, "q8": 8.5}
        avg = sum(share[ty] * bits.get(ty, 0) for ty in share)
        name = f"t{int(round(b * 100))}-" + "-".join(LABEL.get(t, t) for t in types if t != base)
        with open(f"ladder/{name}.choice", "w") as f:
            for u, c in sorted(ch.items()):
                if c[2] != base:
                    for n in U[u]: f.write(f"{n}\t{LABEL[c[2]]}\n")
        print(f"time {b:.2f}: got {tm:.3f}, predicted drift change vs base {h:+.4f}, average {avg:.2f} bits   " +
              "  ".join(f"{ty} {100 * share[ty]:.1f}%" for ty in types) + f"   -> ladder/{name}.choice")
