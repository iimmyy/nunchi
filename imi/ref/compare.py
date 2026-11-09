# How far each set of answers is from the truth (imi answer-key files), per position and overall.
#   python3 compare.py TRUTH.bin NAME=FILE.bin ...
import struct, sys
import numpy as np
def read(p):
    with open(p, "rb") as f:
        ne, nv, n = struct.unpack("3i", f.read(12)); H, L = [], []
        for _ in range(n):
            f.read(4); H.append(np.frombuffer(f.read(4 * ne), np.float32)); L.append(np.frombuffer(f.read(4 * nv), np.float32))
    return np.array(H, np.float64), np.array(L, np.float64)
th, tl = read(sys.argv[1])
print(f"{'answers':18s} {'hidden err mean':>15s} {'max':>8s} {'pos0':>8s} {'score err mean':>15s} {'same word':>10s}")
for arg in sys.argv[2:]:
    name, p = arg.split("=", 1)
    h, l = read(p); n = min(len(h), len(th))
    eh = np.linalg.norm(h[:n] - th[:n], axis=1) / np.linalg.norm(th[:n], axis=1)
    el = np.linalg.norm(l[:n] - tl[:n], axis=1) / np.linalg.norm(tl[:n], axis=1)
    same = int((l[:n].argmax(1) == tl[:n].argmax(1)).sum())
    print(f"{name:18s} {eh.mean():15.2e} {eh.max():8.2e} {eh[0]:8.2e} {el.mean():15.2e} {same:>6d}/{n}")
