// imi.cpp: Seoul's codec (the liquid-codec decoder), codes -> sound, on our own engine.
// Per audio frame: 12 codes -> FSQ level table (4 values each) -> two projections -> one 512-wide vector, repeated to
// 6 positions (6 STFT frames per audio frame) -> 8 LFM layers (5 short conv, 3 gated attention over the last 30
// positions; both carry memory across frames) -> norm -> 641 log magnitudes + 641 phases per position -> inverse FFT
// (1280 points, periodic Hann, hop 320) with overlap-add. 1920 samples at 24 kHz per frame.
#pragma once
#include "../model.h"
#include "tables.h"
#include "team.h"

#include <atomic>
#include <complex>
#include <memory>
#include <string>
#include <vector>

struct imi_codec {
    struct layer {
        const float *ln1, *ln2, *post1, *post2;
        imi_q8g gate_up, down;
        // short conv
        const float * conv_w = nullptr;       // [tap][channel]
        imi_q8g pw1, pw2;
        // attention
        imi_q8g qkv, o;
        std::vector<float> gate_w;            // [n_head][n_embd], float
        const float *q_norm = nullptr, *k_norm = nullptr;
        bool conv = false;
    };
    imi_file f;
    std::vector<layer> layers;
    const float *lvl = nullptr, *qo_w = nullptr, *qo_b = nullptr, *in_b = nullptr, *out_b = nullptr, *out_norm = nullptr;
    imi_q8g in_proj, out_proj;
    int n_lvl = 0, n_cb = 12, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, d_conv = 0;
    int n_fs = 6, window = 30, n_fft = 1280, hop = 320, n_bins = 641;
    float eps = 1e-5f, theta = 1e6f;
    int n_threads = 4;

    bool load(const std::string & path, int threads, imi_team * shared = nullptr);
    void reset();
    // one frame of 12 codes -> appends its samples (1920, less the start padding on the first frames)
    void frame(const int32_t * codes, std::vector<float> & pcm);
    // the end of speech: the tail held back by the overlap
    void flush(std::vector<float> & pcm);
    ~imi_codec();

private:
    struct per_thread { std::vector<float> x, h, xt; };
    imi_team own_team;
    imi_team * team = &own_team;
    std::vector<per_thread> per;
    std::vector<float> big, y, act;                        // shared row-slice outputs
    std::vector<std::vector<float>> conv_state;           // per layer: [d_conv][n_embd]
    std::vector<std::vector<float>> k_win, v_win;         // per layer: last window-1 positions, [window-1][kv_dim]
    std::vector<std::vector<float>> conv_w_t;
    std::vector<float> spec;                              // [n_fs][2 * n_bins]
    int pos = 0;
    std::unique_ptr<std::atomic<int>[]> ctr;
    // inverse FFT + overlap-add
    std::vector<float> hann, ola, wsum;
    std::vector<std::complex<float>> tw;
    int pad_left = 0;
    void run(int tid, const int32_t * codes);
    void istft(const float * logmag_phase, std::vector<float> & pcm);
};
