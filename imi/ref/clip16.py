# One test window for the ears: a slice of a 48 kHz 16-bit WAV, the same loudness step the hearing tests use (-20 dB
# RMS, peak below 0.99), then 48 -> 16 kHz (windowed-sinc low-pass, every third sample). Writes raw float32 samples
# for imi and the 64-bit key, and a 32-bit float WAV at 16 kHz for the fork, so all three hear the same samples.
#   python3 clip16.py WAV48 START_S END_S OUT.f32 OUT.wav
import struct, sys, wave
import numpy as np
wav, s0, s1, out_f32, out_wav = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
with wave.open(wav) as w:
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    raw = w.readframes(n)
assert sr == 48000 and sw == 2, (sr, sw)
y = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
if ch > 1: y = y.reshape(-1, ch).mean(1)
y = y[int(round(s0 * sr)):int(round(s1 * sr))]
rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-9
y = (y * min(10 ** (-20 / 20) / rms, 0.99 / (float(np.abs(y).max()) + 1e-9))).astype(np.float64)
taps = 241
t = np.arange(taps) - (taps - 1) / 2
fc = 7500 / sr
h = 2 * fc * np.sinc(2 * fc * t) * np.kaiser(taps, 8.6)
h /= h.sum()
z = np.convolve(y, h, mode="same")[::3].astype("<f4")
z.tofile(out_f32)
data = z.tobytes()
fmt = struct.pack("<HHIIHH", 3, 1, 16000, 16000 * 4, 4, 32)
with open(out_wav, "wb") as f:
    f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)
print(f"{len(z)} samples at 16 kHz ({len(z) / 16000:.2f} s)")
