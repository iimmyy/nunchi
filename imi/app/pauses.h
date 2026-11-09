// nunchi: the live pause finder (tools/pause_live.py, frame by frame on the live feed). 20 ms frames of 16 kHz sound; a
// frame is quiet when it is quiet_db under the loud level of the last 30 s heard (the 95th percentile frame). A piece
// with sound in it is cut the moment min_pause s of quiet follow at least min_piece s of it; fast speech has only short
// gaps between words, so once a piece holds long_piece s, short_pause s of quiet is enough. A piece with no such gap
// by max_piece s is cut at its quietest 100 ms (at least min_piece s in). While a piece has no sound at all its start
// moves along with the quiet (keeping the last 0.3 s), so silence is never heard as a piece.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

struct nunchi_pauses {
    double min_pause = 0.25, quiet_db = 25, min_piece = 1.0, max_piece = 6.0, short_pause = 0.12, long_piece = 3.0;
    static constexpr int F = 320;          // samples per frame (20 ms)
    struct cut { int64_t start, end; };    // sample positions since the feed began

    // the next samples of the feed; returns the pieces cut by them, oldest first
    std::vector<cut> push(const float * x, size_t n) {
        std::vector<cut> out;
        for (size_t i = 0; i < n; i++) {
            part[np++] = x[i];
            if (np == F) { np = 0; frame(out); }
        }
        return out;
    }
    int64_t piece_start() const { return start * F; }   // the open piece's first sample
    double quiet_s() const { return run * 0.02; }       // how long it has been quiet
    bool has_sound() const { return first_loud >= 0; }  // the open piece holds some sound
    void reset(int64_t at_sample) {                      // start over (the switch went off and on)
        start = done = at_sample / F; first_loud = -1; run = 0; np = 0; lev.clear(); quiet.clear();
    }

private:
    float part[F];
    int np = 0;
    int64_t done = 0, start = 0, first_loud = -1;   // frames: done so far, the open piece's first, its first loud one
    int run = 0;                                    // quiet frames in a row
    std::deque<float> heard;                        // the last 30 s of frame levels
    std::vector<float> lev;                         // the open piece's frame levels, from `start`
    std::vector<char> quiet;
    std::vector<float> tmp;

    void frame(std::vector<cut> & out) {
        double e = 0;
        for (int i = 0; i < F; i++) e += (double) part[i] * part[i];
        const float db = (float) (10 * std::log10(e / F + 1e-12));
        heard.push_back(db);
        if (heard.size() > 1500) heard.pop_front();
        tmp.assign(heard.begin(), heard.end());
        // numpy's 95th percentile (linear between the two nearest frames)
        const double r = 0.95 * (tmp.size() - 1);
        const size_t k = (size_t) r;
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
        double p95 = tmp[k];
        if (k + 1 < tmp.size()) {
            const float nxt = *std::min_element(tmp.begin() + k + 1, tmp.end());
            p95 += (r - k) * (nxt - tmp[k]);
        }
        const bool q = db < p95 - quiet_db;
        const int64_t f = done++;
        lev.push_back(db);
        quiet.push_back(q);
        if (q) run++;
        else { run = 0; if (first_loud < 0) first_loud = f; }
        const int64_t end = f + 1;
        if (first_loud < 0 && end - start > 15) {   // no sound yet: the start keeps up with the quiet
            const int64_t drop = end - 15 - start;
            lev.erase(lev.begin(), lev.begin() + drop);
            quiet.erase(quiet.begin(), quiet.begin() + drop);
            start += drop;
            return;
        }
        const double said = (end - run - start) * 0.02;   // the piece up to its quiet
        if (first_loud >= 0 && ((run * 0.02 >= min_pause && said >= min_piece) || (run * 0.02 >= short_pause - 1e-9 && said >= long_piece))) {
            out.push_back({ start * F, end * F });
            start = end; first_loud = -1; run = 0; lev.clear(); quiet.clear();
        } else if ((end - start) * 0.02 >= max_piece) {
            const int64_t L = end - start, lo = (int64_t) (min_piece / 0.02);
            int64_t c = end;
            if (L > lo) {   // the quietest 100 ms (5 frames, numpy's "same" with zeros past the ends)
                double best = 1e30;
                for (int64_t i = lo; i < L; i++) {
                    double s = 0;
                    for (int64_t j = i - 2; j <= i + 2; j++) if (j >= 0 && j < L) s += lev[j];
                    if (s / 5 < best) { best = s / 5; c = start + i; }
                }
            }
            out.push_back({ start * F, c * F });
            const int64_t drop = c - start;
            lev.erase(lev.begin(), lev.begin() + drop);
            quiet.erase(quiet.begin(), quiet.begin() + drop);
            start = c;
            first_loud = -1; run = 0;
            for (int64_t g = 0; g < (int64_t) quiet.size(); g++) if (!quiet[g]) { first_loud = start + g; break; }
            for (int64_t g = (int64_t) quiet.size() - 1; g >= 0 && quiet[g]; g--) run++;
        }
    }
};
