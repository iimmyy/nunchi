// nunchi: sound from any rate to 16 kHz as it streams in (windowed sinc, 32 taps, the cutoff at the lower rate's
// half, so nothing above 8 kHz folds back into the speech)
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

struct nunchi_resampler {
    void reset(double in_rate, double out_rate) {
        step = in_rate / out_rate;
        fc = std::min(1.0, out_rate / in_rate) * 0.95;
        buf.assign(TAPS, 0.0f);   // the past, silent
        t = TAPS;
    }
    void push(const float * x, size_t n, std::vector<float> & out) {
        if (step == 1.0) { out.insert(out.end(), x, x + n); return; }
        buf.insert(buf.end(), x, x + n);
        while (t + TAPS < (double) buf.size()) {
            const long c = (long) t;
            double s = 0, w = 0;
            for (long k = c - TAPS + 1; k <= c + TAPS; k++) {
                const double d = t - k, a = d * fc;
                const double sinc = std::fabs(a) < 1e-9 ? 1.0 : std::sin(3.14159265358979323846 * a) / (3.14159265358979323846 * a);
                const double win = 0.5 + 0.5 * std::cos(3.14159265358979323846 * d / (TAPS + 1));
                const double h = sinc * win;
                s += buf[(size_t) k] * h;
                w += h;
            }
            out.push_back((float) (s / w));
            t += step;
        }
        const long drop = (long) t - TAPS;
        if (drop > 0) { buf.erase(buf.begin(), buf.begin() + drop); t -= drop; }
    }
    double step = 1.0;

private:
    static constexpr int TAPS = 16;
    double fc = 1.0, t = TAPS;
    std::vector<float> buf;
};
