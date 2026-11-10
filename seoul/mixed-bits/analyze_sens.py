"""Read the whole-brain test of each table (sens/sens.tsv) next to its local output error (pieces/tables.tsv).
dKL_T = KL(brain with only table T at 2-bit) - KL(all 4-bit brain), on 8 held-out Korean sermon subtitles.
factor_T = dKL_T / (local 2-bit error - local 4-bit error): how much whole-brain drift one unit of this table's
output error costs. Also: do the singles add up to the all-2-bit brain, and do the 3-bit checks follow the factor."""
import csv, collections, sys
sens = {}
for r in csv.reader(open("sens/sens.tsv"), delimiter="\t"):
    sens[r[0]] = (float(r[1]), float(r[2]))
tab = {r["table"]: r for r in csv.DictReader(open("pieces/tables.tsv"), delimiter="\t")}
base = sens["all-Q4_K"][0]
# 2026-11-10: the single-table brains were made on the box; the box's own all-4-bit brain is not byte-identical to
# the Mac-made donor (KL 0.1014 vs 0.1035, base2/base.tsv), so the singles are measured against the box's.
BOX_Q4 = 0.101398
print(f"Mac-made all-Q4_K {base:.4f}; box-made {BOX_Q4:.4f} (used as the base)")
base = BOX_Q4
print(f"all-Q4_K KL {base:.4f} top {sens['all-Q4_K'][1]:.1f}%   all-Q3_K {sens['all-Q3_K'][0]:.4f}   all-Q2_K {sens['all-Q2_K'][0]:.4f}")
rows = []
for name, r in tab.items():
    k = name + ":q2_K"
    if k not in sens: continue
    e4, e3, e2 = (float(r[f"{t}_true"]) for t in ("Q4_K", "Q3_K", "Q2_K"))
    d = sens[k][0] - base
    rows.append((name, d, e2 - e4, d / (e2 - e4), e3 - e4, int(r["rows"]) * int(r["cols"])))
tot = sum(x[1] for x in rows)
print(f"{len(rows)} tables; sum of single 2-bit dKL {tot:.4f} vs all-2-bit brain {sens['all-Q2_K'][0]-base:.4f}")
# 3-bit checks
print("\n3-bit checks: measured dKL vs factor x local 3-bit error")
for name, d, de2, f, de3, n in rows:
    k = name + ":q3_K"
    if k in sens: print(f"  {name:40s} 2-bit dKL {d:+.4f}   3-bit dKL {sens[k][0]-base:+.4f}  predicted {f*de3:+.4f}")
# by kind
kind = collections.defaultdict(list)
for x in rows: kind[x[0].split(".", 2)[2].replace(".weight", "")].append(x)
print("\nby table kind: mean dKL, share of params")
P = sum(x[5] for x in rows)
for k, v in sorted(kind.items(), key=lambda kv: -sum(x[1] for x in kv[1])):
    print(f"  {k:22s} n {len(v):2d}  sum dKL {sum(x[1] for x in v):+.4f}  mean {sum(x[1] for x in v)/len(v):+.4f}  params {100*sum(x[5] for x in v)/P:4.1f}%")
print("\nby layer: sum dKL")
lay = collections.defaultdict(float)
for x in rows: lay[int(x[0].split(".")[1])] += x[1]
print("  " + " ".join(f"L{l}:{v:+.3f}" for l, v in sorted(lay.items())))
print("\ntop 20 tables by dKL")
for x in sorted(rows, key=lambda x: -x[1])[:20]: print(f"  {x[0]:40s} dKL {x[1]:+.4f}  per-param x1e9 {1e9*x[1]/x[5]:.1f}")
print("\nbottom 10")
for x in sorted(rows, key=lambda x: x[1])[:10]: print(f"  {x[0]:40s} dKL {x[1]:+.4f}")
neg = [x for x in rows if x[1] < 0.002]
print(f"\ntables with dKL < 0.002 (within the noise of the 3-bit checks): {len(neg)}, params {100*sum(x[5] for x in neg)/P:.1f}%")
