# Our voice's 24 kHz WAV -> 16 kHz float32 chunks for imi-hear (up 2, windowed-sinc low-pass, every third sample).
#   python3 voice16.py IN.wav OUT_PREFIX [CHUNK_S]
import sys, wave
import numpy as np
w = wave.open(sys.argv[1]); assert w.getframerate() == 24000 and w.getsampwidth() == 2
y = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
up = np.zeros(2 * len(y)); up[::2] = 2 * y
taps = 241; t = np.arange(taps) - (taps - 1) / 2; fc = 7500 / 48000
h = 2 * fc * np.sinc(2 * fc * t) * np.kaiser(taps, 8.6); h /= h.sum()
z = np.convolve(up, h, mode="same")[::3].astype("<f4")
step = int(float(sys.argv[3]) * 16000) if len(sys.argv) > 3 else len(z)
for i, s in enumerate(range(0, len(z), step)):
    z[s:s + step].tofile(f"{sys.argv[2]}-{i}.f32")
    print(f"{sys.argv[2]}-{i}.f32 {s / 16000:.1f}-{min(len(z), s + step) / 16000:.1f} s")
