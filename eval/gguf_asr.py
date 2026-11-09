"""Seoul GGUF ASR on the same windows as a PyTorch run, through a running llama-server (OpenAI chat API).

  python3 gguf_asr.py REF.jsonl WAV48 OUT.jsonl [--port 8090] [--ir IRS.npy:IDX]

REF.jsonl: a widen_seoul.py asr output (its window start/end are reused, so the scores line up window for window).
WAV48: the same 48 kHz test file the PyTorch run read. Each window gets the same loudness step (loud(), -20 dB RMS).
Writes loud_ko like widen_seoul.py, plus seconds, prompt/answer token counts and the server's own timings.
"""
import base64, json, struct, sys, time, urllib.request, wave
import numpy as np

ASR = "Perform ASR in Korean."


def read_wav(path):
    with wave.open(path) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    assert sw == 2, "16-bit WAV expected"
    y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        y = y.reshape(-1, ch).mean(1)
    return sr, y


def loud(y, target_db=-20.0):  # same as widen_seoul.loud
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-9
    g = min(10 ** (target_db / 20) / rms, 0.99 / (float(np.abs(y).max()) + 1e-9))
    return (y * g).astype(np.float32)


def float_wav(y, sr):  # 32-bit float WAV, so nothing is rounded to 16 bits on the way in
    data = y.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE" + b"fmt " +
            struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data)


def ask(port, wav_bytes, max_tokens=512):
    body = {"messages": [{"role": "system", "content": ASR},
                         {"role": "user", "content": [{"type": "input_audio", "input_audio": {
                             "data": base64.b64encode(wav_bytes).decode(), "format": "wav"}}]}],
            "temperature": 0, "max_tokens": max_tokens, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}   # no thinking, as in PyTorch and the product
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main():
    ref, wav48, out = sys.argv[1:4]
    rest = sys.argv[4:]
    port = int(rest[rest.index("--port") + 1]) if "--port" in rest else 8090
    sr, y = read_wav(wav48)
    assert sr == 48000, sr
    if "--ir" in rest:
        path, idx = rest[rest.index("--ir") + 1].rsplit(":", 1)
        ir = np.load(path)[int(idx)]
        from numpy.fft import rfft, irfft
        n = len(y) + len(ir) - 1
        y = irfft(rfft(y, n) * rfft(ir, n), n)[: len(y)].astype(np.float32)
    rows = [json.loads(l) for l in open(ref, encoding="utf-8")]
    import os
    done = {json.loads(l)["window"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    with open(out, "a", encoding="utf-8") as fo:   # picks up where an earlier run stopped
        for r in rows:
            if r["window"] in done:
                continue
            s, e = int(round(r["start"] * sr)), int(round(r["end"] * sr))
            clip = loud(y[s:e])
            t = time.time()
            res = ask(port, float_wav(clip, sr))
            dt = time.time() - t
            text = res["choices"][0]["message"]["content"].strip()
            u = res.get("usage", {})
            rec = {"window": r["window"], "start": r["start"], "end": r["end"], "side": None, "loud_ko": text,
                   "ran_out": res["choices"][0].get("finish_reason") == "length", "s": round(dt, 2),
                   "prompt_tokens": u.get("prompt_tokens"), "answer_tokens": u.get("completion_tokens"),
                   "timings": res.get("timings")}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            print(r["window"], round(dt, 1), text[:60], flush=True)


if __name__ == "__main__":
    main()
