# Listen file: the speaker's Korean (quieter) with the English as it would be heard live, on the same clock.
#   python3 live_mix.py SERMON.wav START_S END_S LIVE.wav OUT.wav [korean_db=-12]
# SERMON.wav any rate/channels (16-bit PCM); LIVE.wav 24 kHz mono from imi-instant (time 0 = START_S). OUT 24 kHz mono.
import sys, wave
import numpy as np
def read(p):
    w = wave.open(p, "rb"); n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
    x = np.frombuffer(w.readframes(n), "<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(1), sr
ko, sr = read(sys.argv[1]); s0, s1 = float(sys.argv[2]), float(sys.argv[3])
ko = ko[int(s0 * sr): int(s1 * sr)]
if sr != 24000:   # plain linear resampling, good enough to listen
    t = np.arange(int(len(ko) * 24000 / sr)) * sr / 24000.0
    ko = np.interp(t, np.arange(len(ko)), ko).astype(np.float32)
en, _ = read(sys.argv[4])
g = 10 ** ((float(sys.argv[6]) if len(sys.argv) > 6 else -12.0) / 20)
n = max(len(ko), len(en))
mix = np.zeros(n, np.float32); mix[:len(ko)] += g * ko; mix[:len(en)] += en
mix /= max(1.0, np.abs(mix).max() / 0.98)
o = wave.open(sys.argv[5], "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(24000)
o.writeframes((mix * 32767).astype("<i2").tobytes()); o.close()
print(f"{sys.argv[5]}: {n / 24000:.1f} s (Korean {len(ko) / 24000:.1f} s at {20 * np.log10(g):.0f} dB, English {len(en) / 24000:.1f} s)")
