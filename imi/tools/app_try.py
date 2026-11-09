# Trying the app without a browser: plays a WAV into nunchi's /mic the way the screen page does (16-bit chunks every
# 100 ms at the file's own rate, so the app's resampler runs), and listens to /events like a phone, writing every
# event with the time it arrived.
#   python3 app_try.py http://localhost:8080 RECORDING.wav FROM_S SECONDS OUT.tsv [audio=1]
# OUT.tsv: arrival time (s since the sound started) \t event \t its data (speech chunks: their sample count)
import base64, json, sys, threading, time, urllib.request, wave
url, path, start, secs, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
want_audio = len(sys.argv) <= 6 or sys.argv[6] == "1"
w = wave.open(path, "rb")
rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
assert sw == 2, "16-bit WAV only"
w.setpos(int(start * rate))
t0 = None
rows = []
lock = threading.Lock()

def listen():
    r = urllib.request.urlopen(url + "/events" + ("?audio=1" if want_audio else ""), timeout=600)
    ev, data = None, ""
    for raw in r:
        line = raw.decode("utf-8").rstrip("\n")
        if line.startswith("event: "): ev = line[7:]
        elif line.startswith("data: "): data = line[6:]
        elif line == "" and ev:
            now = time.time() - t0 if t0 else -1
            m = json.loads(data)
            if ev == "sp": m = {"id": m["id"], "samples": len(base64.b64decode(m["d"])) // 2}
            with lock: rows.append((now, ev, json.dumps(m, ensure_ascii=False)))
            ev, data = None, ""

threading.Thread(target=listen, daemon=True).start()
time.sleep(1.0)
urllib.request.urlopen(urllib.request.Request(url + "/api/set?listening=1", method="POST")).read()
n = int(rate * 0.1)
t0 = time.time()
k = 0
while k * 0.1 < secs:
    b = w.readframes(n)
    if not b: break
    if ch > 1:   # to mono: the first channel (the page sends one channel)
        import array
        a = array.array("h", b)
        b = array.array("h", a[::ch]).tobytes()
    urllib.request.urlopen(urllib.request.Request(f"{url}/mic?rate={rate}", data=b, method="POST")).read()
    k += 1
    time.sleep(max(0, t0 + k * 0.1 - time.time()))
time.sleep(25)   # the last sentences finish
with lock, open(out, "w") as f:
    for r in rows: f.write(f"{r[0]:.2f}\t{r[1]}\t{r[2]}\n")
print(f"{out}: {len(rows)} events, {sum(1 for r in rows if r[1] == 'en')} sentences")
