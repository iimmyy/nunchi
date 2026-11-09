// nunchi: the live Korean to English interpreter for a room. One program on the room's computer: it hears the sound
// the screen page sends (the computer's own input: the mixer, a microphone), runs Seoul, and serves the page every
// phone on the same Wi-Fi opens. Nothing goes to the internet.
//   nunchi [--model seoul.gguf] [--mm seoul-mm.gguf] [--port 8080] [--threads N] [--no-voice] [--no-browser]
//          [--terms terms.txt] [--wav RECORDING.wav [--from SECONDS]]
//   --wav plays a recording in as if it were live (for trying it out without a microphone)
#include "ggml.h"
#include "live.h"
#include "pages.h"
#include "qr.h"
#include "resample.h"
#include "server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include "win_compat.h"
#include <mmsystem.h>
#include <shellapi.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

std::vector<std::string> nunchi_addresses();

// Files by UTF-8 name on every system. Windows' plain file calls (fopen, std::ifstream) read a name in the computer's
// old code page, so a folder with Korean letters in its path (a Korean user name puts them in the Desktop's path) would
// not open. ggml_fopen hands the name to Windows' wide-character call instead (elsewhere it is plain fopen).
static FILE * open_utf8(const std::string & p, const char * mode) { return ggml_fopen(p.c_str(), mode); }
static bool exists(const std::string & p) {
    FILE * f = open_utf8(p, "rb");
    if (f) fclose(f);
    return f != nullptr;
}
static std::string read_file(const std::string & p) {
    std::string s;
    if (FILE * f = open_utf8(p, "rb")) {
        char b[65536];
        size_t n;
        while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
        fclose(f);
    }
    return s;
}
static bool write_file(const std::string & p, const std::string & s) {
    FILE * f = open_utf8(p, "wb");
    if (!f) return false;
    const bool ok = fwrite(s.data(), 1, s.size(), f) == s.size();
    return fclose(f) == 0 && ok;
}
#ifdef _WIN32
static std::string utf8(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 1 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}
#endif

static std::atomic<bool> g_stop{ false };
static void on_signal(int) { g_stop = true; }

static std::string exe_dir(const char * argv0) {
    std::string p;
#ifdef _WIN32
    std::vector<wchar_t> b(32768);
    if (GetModuleFileNameW(nullptr, b.data(), (DWORD) b.size())) p = utf8(b.data());
#elif defined(__APPLE__)
    char b[4096];
    uint32_t n = sizeof b;
    if (_NSGetExecutablePath(b, &n) == 0) p = b;
#else
    char b[4096];
    const ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    if (n > 0) p.assign(b, (size_t) n);
#endif
    if (p.empty()) p = argv0;
    const size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? "." : p.substr(0, s);
}


static void open_browser(const std::string & url) {
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    std::system(("open '" + url + "'").c_str());
#else
    std::system(("xdg-open '" + url + "' >/dev/null 2>&1 &").c_str());
#endif
}

// a WAV file (16-bit or float PCM, any channels) as mono samples, and its rate
static bool read_wav(const std::string & path, std::vector<float> & out, int & rate) {
    const std::string d = read_file(path);
    if (d.size() < 12 || memcmp(d.data(), "RIFF", 4) || memcmp(d.data() + 8, "WAVE", 4)) return false;
    int fmt = 0, ch = 0, bits = 0;
    for (size_t i = 12; i + 8 <= d.size();) {
        uint32_t n;
        memcpy(&n, d.data() + i + 4, 4);
        const char * c = d.data() + i + 8;
        if (!memcmp(d.data() + i, "fmt ", 4) && n >= 16) {
            uint16_t f16, ch16, b16; uint32_t r;
            memcpy(&f16, c, 2); memcpy(&ch16, c + 2, 2); memcpy(&r, c + 4, 4); memcpy(&b16, c + 14, 2);
            fmt = f16; ch = ch16; rate = (int) r; bits = b16;
            if (fmt == 0xFFFE && n >= 26) { uint16_t sub; memcpy(&sub, c + 24, 2); fmt = sub; }
        } else if (!memcmp(d.data() + i, "data", 4) && ch > 0) {
            const size_t frames = std::min<size_t>(n, d.size() - i - 8) / (size_t) (ch * bits / 8);
            out.resize(frames);
            for (size_t k = 0; k < frames; k++) {
                double s = 0;
                for (int j = 0; j < ch; j++) {
                    const char * p = c + (k * ch + j) * (bits / 8);
                    if (fmt == 1 && bits == 16) { int16_t v; memcpy(&v, p, 2); s += v / 32768.0; }
                    else if (fmt == 3 && bits == 32) { float v; memcpy(&v, p, 4); s += v; }
                    else if (fmt == 1 && bits == 24) { const int v = (int) ((uint8_t) p[0] << 8 | (uint8_t) p[1] << 16 | (uint32_t) (uint8_t) p[2] << 24) >> 8; s += v / 8388608.0; }
                    else return false;
                }
                out[k] = (float) (s / ch);
            }
            return true;
        }
        i += 8 + n + (n & 1);
    }
    return false;
}


#ifdef _WIN32
// Windows, for as long as the program runs: the computer must not sleep or turn its screen off in the middle of a
// sermon, must not slow Nunchi down to save power when its window is behind the screen page (Windows 11's EcoQoS),
// and must wake its waits on time (Windows' timer ticks every 15.6 ms unless a program asks for 1 ms; all of it
// ends with the program).
static void windows_full_speed() {
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
    timeBeginPeriod(1);
    PROCESS_POWER_THROTTLING_STATE pt = {};
    pt.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    pt.StateMask   = 0;
    if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof(pt))) return;
    // Windows 10 has no timer-resolution throttling and refuses the whole request when it's asked for (error 87)
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof(pt)))
        printf("(Windows did not take the full-speed setting, error %lu; Nunchi still runs)\n", (unsigned long) GetLastError());
}
#endif

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    windows_full_speed();
#endif
    // the arguments as UTF-8 (on Windows argv comes in the old code page; the wide command line has every letter)
    std::vector<std::string> args(argv, argv + argc);
#ifdef _WIN32
    int wn = 0;
    if (wchar_t ** w = CommandLineToArgvW(GetCommandLineW(), &wn)) {
        args.clear();
        for (int i = 0; i < wn; i++) args.push_back(utf8(w[i]));
        LocalFree(w);
    }
#endif
    argc = (int) args.size();
    const std::string dir = exe_dir(args.empty() ? "" : args[0].c_str());
    std::string model, mm, wav, terms_file;
    // every processor but one: the browser showing the screen page (and taking the microphone) and Windows' own work
    // need one, and a team that takes them all is slowed down whenever anything else runs
    int port = 8080, threads = (int) std::thread::hardware_concurrency() - 1;
    double from = 0;
    bool voice = true, browser = true;
    for (int i = 1; i < argc; i++) {
        const std::string a = args[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? args[++i] : ""; };
        if (a == "--model") model = val();
        else if (a == "--mm") mm = val();
        else if (a == "--port") port = atoi(val().c_str());
        else if (a == "--threads") threads = atoi(val().c_str());
        else if (a == "--wav") wav = val();
        else if (a == "--from") from = atof(val().c_str());
        else if (a == "--terms") terms_file = val();
        else if (a == "--no-voice") voice = false;
        else if (a == "--no-browser") browser = false;
        else { fprintf(stderr, "nunchi: unknown option %s\n", a.c_str()); return 1; }
    }
    if (threads < 1) threads = 4;
    auto find = [&](const std::string & given, const char * name) {
        if (!given.empty()) return given;
        for (const std::string & d : { dir, dir + "/models", std::string("."), std::string("models") })
            if (exists(d + "/" + name)) return d + "/" + name;
        return std::string(name);
    };
    model = find(model, "seoul.gguf");
    mm = find(mm, "seoul-mm.gguf");
    if (terms_file.empty() && exists(dir + "/terms.txt")) terms_file = dir + "/terms.txt";

    printf("Nunchi: live Korean to English\n");
    if (!exists(model) || !exists(mm)) {
        printf("The model files are missing. Put seoul.gguf and seoul-mm.gguf next to this program.\n  looked for %s\n  and %s\n", model.c_str(), mm.c_str());
        printf("Press Enter to close.\n");
        getchar();
        return 1;
    }
    printf("Loading Seoul (%d threads)...\n", threads);
    const auto tl = std::chrono::steady_clock::now();
    nunchi_live live;
    if (!live.load(model, mm, threads)) { printf("Seoul could not load. Press Enter to close.\n"); getchar(); return 1; }
    printf("Loaded in %.1f s.\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
    live.voice_on = voice;
    if (!terms_file.empty()) live.set_terms(read_file(terms_file));

    nunchi_server server;
    std::mutex rs_mu;
    nunchi_resampler rs;
    double rs_rate = 0;
    std::mutex log_mu;
    std::vector<std::string> logs;
    server.handle = [&](const nunchi_request & r) {
        nunchi_reply rep;
        if (r.path == "/" || r.path == "/index.html") rep.body = NUNCHI_PHONE;
        else if (r.path == "/screen") rep.body = NUNCHI_SCREEN;
        else if (r.path == "/worklet.js") { rep.type = "text/javascript"; rep.body = NUNCHI_WORKLET; }
        else if (r.path == "/qr.svg") { rep.type = "image/svg+xml"; rep.body = nunchi_qr_svg(r.q("text")); if (rep.body.empty()) rep.code = 404; }
        else if (r.path == "/mic" && r.method == "POST") {
            const double rate = atof(r.q("rate").c_str());
            const size_t n = r.body.size() / 2;
            std::vector<float> x(n), y;
            for (size_t i = 0; i < n; i++) { int16_t v; memcpy(&v, r.body.data() + 2 * i, 2); x[i] = v / 32768.0f; }
            {
                std::lock_guard<std::mutex> l(rs_mu);
                if (rate > 0 && rate != rs_rate) { rs.reset(rate, 16000); rs_rate = rate; }
                rs.push(x.data(), x.size(), y);
            }
            live.push(y.data(), y.size());
            rep.type = "text/plain";
            rep.body = "ok";
        } else if (r.path == "/api/set" && r.method == "POST") {
            if (!r.q("listening").empty()) live.listening = r.q("listening") == "1";
            if (!r.q("voice").empty()) live.voice_on = r.q("voice") == "1";
            rep.type = "text/plain";
            rep.body = "ok";
        } else if (r.path == "/api/terms" && r.method == "POST") {
            live.set_terms(r.body);
            if (terms_file.empty()) terms_file = dir + "/terms.txt";
            write_file(terms_file, r.body);
            rep.type = "text/plain";
            rep.body = "ok";
        } else if (r.path == "/api/state") {
            std::string a = "[";
            for (auto & s : nunchi_addresses()) a += (a.size() > 1 ? "," : "") + nunchi_json(s);
            rep.type = "application/json";
            rep.body = "{\"listening\":" + std::string(live.listening ? "true" : "false") + ",\"voice\":" + (live.voice_on ? "true" : "false") +
                       ",\"listeners\":" + std::to_string(server.listeners()) + ",\"voice_listeners\":" + std::to_string(server.voice_listeners()) +
                       ",\"pieces\":" + std::to_string(live.heard_pieces.load()) + ",\"sentences\":" + std::to_string(live.sentences.load()) +
                       ",\"level\":" + std::to_string(live.level_db.load()) + ",\"port\":" + std::to_string(port) + ",\"addresses\":" + a +
                       "],\"terms\":" + nunchi_json(live.terms_text()) + "}";
        } else if (r.path == "/api/log") {
            std::lock_guard<std::mutex> l(log_mu);
            rep.type = "text/plain; charset=utf-8";
            for (auto & s : logs) rep.body += s + "\n";
        } else { rep.code = 404; rep.body = "not here"; }
        return rep;
    };
    std::string err;
    if (!server.start(port, err)) { printf("Nunchi could not start its page: %s. Press Enter to close.\n", err.c_str()); getchar(); return 1; }

    const auto t0 = std::chrono::steady_clock::now();
    nunchi_events ev;
    ev.heard = [&](const std::string & ko) { server.send("ko", "{\"ko\":" + nunchi_json(ko) + "}"); };
    ev.writing = [&](int id, const std::string & en) { server.send("w", "{\"id\":" + std::to_string(id) + ",\"en\":" + nunchi_json(en) + "}"); };
    ev.said = [&](int id, const std::string & ko, const std::string & en) {
        server.send("en", "{\"id\":" + std::to_string(id) + ",\"ko\":" + nunchi_json(ko) + ",\"en\":" + nunchi_json(en) + "}", true);
    };
    ev.speech = [&](int id, const float * pcm, size_t n) {
        std::vector<uint8_t> b(n * 2);
        for (size_t i = 0; i < n; i++) {
            const float v = std::max(-1.0f, std::min(1.0f, pcm[i]));
            const int16_t s = (int16_t) lrintf(v * 32767.0f);
            memcpy(&b[2 * i], &s, 2);
        }
        server.send("sp", "{\"id\":" + std::to_string(id) + ",\"d\":\"" + nunchi_base64(b.data(), b.size()) + "\"}", false, true);
    };
    ev.log = [&](const std::string & line) {
        char ts[32];
        snprintf(ts, sizeof ts, "%8.2f  ", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const std::string s = ts + line;
        printf("%s\n", s.c_str());
        fflush(stdout);
        std::lock_guard<std::mutex> l(log_mu);
        logs.push_back(s);
        if (logs.size() > 400) logs.erase(logs.begin(), logs.begin() + 100);
    };
    std::thread engine([&] { live.run(ev, g_stop); });

    std::thread player;
    if (!wav.empty()) {
        std::vector<float> x;
        int rate = 16000;
        if (!read_wav(wav, x, rate)) { printf("Can't read %s (16-bit, 24-bit or float WAV).\n", wav.c_str()); g_stop = true; }
        else {
            // converted to 16 kHz as it plays, 100 ms at a time, the way the microphone's sound comes in (converting
            // the whole file first held the start back by seconds: 200 s took 10 s on a Windows test machine)
            const size_t a = std::min(x.size(), (size_t) (from * rate));
            printf("Playing %s from %.1f s (%.1f s) in as if live.\n", wav.c_str(), from, (x.size() - a) / (double) rate);
            player = std::thread([&live, x = std::move(x), rate, a] {
                nunchi_resampler r;
                r.reset(rate, 16000);
                std::vector<float> y;
                const size_t chunk = (size_t) rate / 10;
                const auto start = std::chrono::steady_clock::now();
                for (size_t i = a, j = 0; i < x.size() && !g_stop; i += chunk, j++) {
                    std::this_thread::sleep_until(start + std::chrono::milliseconds(100 * (j + 1)));
                    y.clear();
                    r.push(x.data() + i, std::min(chunk, x.size() - i), y);
                    if (!y.empty()) live.push(y.data(), y.size());
                }
            });
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    const std::vector<std::string> addrs = nunchi_addresses();
    printf("\nNunchi is running.\n  This computer: http://localhost:%d/screen\n", port);
    for (auto & a : addrs) printf("  Phones on the same Wi-Fi: http://%s:%d\n", a.c_str(), port);
    printf("Close this window to stop.\n\n");
    fflush(stdout);
    if (browser && wav.empty()) open_browser("http://localhost:" + std::to_string(port) + "/screen");
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (player.joinable()) player.join();
    engine.join();
    server.stop();
    return 0;
}
