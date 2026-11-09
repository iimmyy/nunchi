# The live pause finder: cuts a 16 kHz float32 clip at the speaker's pauses using only the sound already in, frame by
# frame, the way it runs on a live feed. 20 ms frames; a frame is quiet when its level is quiet_db under the loud level
# of the last mem s heard (the 95th percentile frame, with PRE.f32, the sound just before the clip, counted as heard).
# A piece with sound in it is cut the moment min_pause s of quiet follow at least min_piece s of it: the cut is
# then and there, so the piece is in at once. Fast speech has only short gaps between words: once a piece holds
# long_piece s, short_pause s of quiet is enough. A piece with no such gap by max_piece s is cut at its quietest 100 ms
# (at least min_piece s in), known at that moment. The clip's end closes the last piece.
#   python3 pause_live.py CLIP.f32 OUT_PREFIX [PRE.f32|-] [min_pause=0.25] [quiet_db=25] [min_piece=1.0] [max_piece=6.0]
#                         [mem=30] [short_pause=0.12] [long_piece=3.0]
#   writes OUT_PREFIX-NN.f32 and prints "NN start end ready" per piece (ready: when the cut is known); OUT_PREFIX "-"
#   only prints
import sys
from collections import deque
import numpy as np
a = sys.argv[4:]
min_pause = float(a[0]) if len(a) > 0 else 0.25
quiet_db = float(a[1]) if len(a) > 1 else 25.0
min_piece = float(a[2]) if len(a) > 2 else 1.0
max_piece = float(a[3]) if len(a) > 3 else 6.0
mem = float(a[4]) if len(a) > 4 else 30.0
short_pause = float(a[5]) if len(a) > 5 else 0.12
long_piece = float(a[6]) if len(a) > 6 else 3.0
F, FS = 320, 0.02
def levels(x):
    n = len(x) // F
    return 10 * np.log10(np.mean(x[: n * F].reshape(n, F) ** 2, 1) + 1e-12)
y = np.fromfile(sys.argv[1], "<f4").astype(np.float64)
pre = np.fromfile(sys.argv[3], "<f4").astype(np.float64) if len(sys.argv) > 3 and sys.argv[3] != "-" else np.zeros(0)
db = levels(y)
n = len(db)
W = int(round(mem / 0.02))
heard = deque(levels(pre)[-W:], maxlen=W)   # the last mem seconds
quiet = np.zeros(n, bool)
pieces = []   # (start frame, end frame, ready frame)
start, first_loud, run = 0, None, 0
for f in range(n):
    heard.append(db[f])
    quiet[f] = db[f] < np.percentile(heard, 95) - quiet_db
    if quiet[f]: run += 1
    else:
        run = 0
        if first_loud is None: first_loud = f
    end = f + 1
    said = (end - run - start) * FS   # the piece up to its quiet
    if first_loud is not None and ((run * FS >= min_pause and said >= min_piece) or (run * FS >= short_pause and said >= long_piece)):
        pieces.append((start, end, end))
        start, first_loud, run = end, None, 0
    elif (end - start) * FS >= max_piece:
        lo = start + int(min_piece / FS)
        e = np.convolve(db[start:end], np.ones(5) / 5, "same")
        c = lo + int(np.argmin(e[lo - start:])) if end > lo else end
        pieces.append((start, c, end))
        start = c
        loud = [g for g in range(c, end) if not quiet[g]]   # the new piece's frames already in
        first_loud = loud[0] if loud else None
        run = 0
        for g in range(end - 1, c - 1, -1):
            if not quiet[g]: break
            run += 1
if start < n and (not quiet[start:].all() or not pieces):   # only quiet left after the last cut: nothing to hear
    pieces.append((start, n, n))
for k, (s, e, r) in enumerate(pieces):
    print(f"{k:02d} {s * FS:.2f} {e * FS:.2f} {r * FS:.2f}")
    if sys.argv[2] != "-": y[s * F: e * F if e < n else len(y)].astype("<f4").tofile(f"{sys.argv[2]}-{k:02d}.f32")
