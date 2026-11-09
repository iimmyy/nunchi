# Cut a 16 kHz float32 clip at the speaker's pauses: 20 ms frames, a frame is quiet when its level is under
# quiet_db below the clip's loud level (95th percentile frame); a pause is a run of quiet frames of at least min_pause s.
# Each piece ends in the middle of a pause. Pieces shorter than min_piece s join the next one; a piece longer than
# max_piece s is cut at its longest pause inside, or at its quietest 100 ms if it has none.
#   python3 pause_cut.py CLIP.f32 OUT_PREFIX [min_pause=0.25] [quiet_db=25] [min_piece=1.0] [max_piece=6.0]
#   writes OUT_PREFIX-NN.f32 and prints "NN start end" per piece; OUT_PREFIX "-" only prints
import sys
import numpy as np
a = sys.argv[3:]
min_pause = float(a[0]) if len(a) > 0 else 0.25
quiet_db = float(a[1]) if len(a) > 1 else 25.0
min_piece = float(a[2]) if len(a) > 2 else 1.0
max_piece = float(a[3]) if len(a) > 3 else 6.0
y = np.fromfile(sys.argv[1], "<f4").astype(np.float64)
F = 320
n = len(y) // F
db = 10 * np.log10(np.mean(y[: n * F].reshape(n, F) ** 2, 1) + 1e-12)
quiet = db < np.percentile(db, 95) - quiet_db
runs, i = [], 0   # pauses: (first frame, frames)
while i < n:
    if quiet[i]:
        j = i
        while j < n and quiet[j]: j += 1
        if (j - i) * 0.02 >= min_pause: runs.append((i, j - i))
        i = j
    else: i += 1
cuts = [0] + [r0 + r // 2 for r0, r in runs if 0 < r0 and r0 + r < n] + [n]
def split(a, b):   # frames a..b, a long piece cut at its longest pause, else its quietest 100 ms
    if (b - a) * 0.02 <= max_piece: return [(a, b)]
    inner = [(r, r0 + r // 2) for r0, r in runs if a < r0 + r // 2 < b and (r0 + r // 2 - a) * 0.02 >= min_piece and (b - r0 - r // 2) * 0.02 >= min_piece]
    if inner: c = max(inner)[1]
    else:
        lo, hi = a + int(min_piece / 0.02), b - int(min_piece / 0.02)
        e = np.convolve(db, np.ones(5) / 5, "same")
        c = lo + int(np.argmin(e[lo:hi])) if hi > lo else (a + b) // 2
    return split(a, c) + split(c, b)
pieces, start = [], 0
for c in cuts[1:]:
    if (c - start) * 0.02 < min_piece and c != n: continue
    pieces += split(start, c)
    start = c
if len(pieces) > 1 and (pieces[-1][1] - pieces[-1][0]) * 0.02 < min_piece:
    pieces[-2] = (pieces[-2][0], pieces[-1][1]); pieces.pop()
for k, (s, e) in enumerate(pieces):
    print(f"{k:02d} {s * 0.02:.2f} {e * 0.02:.2f}")
    if sys.argv[2] != "-": y[s * F: e * F if e < n else len(y)].astype("<f4").tofile(f"{sys.argv[2]}-{k:02d}.f32")
