// imi.cpp: Seoul's ears (the LFM3-Audio encoder) on our own engine.
//   16 kHz sound -> 128 mel bands every 10 ms -> three stride-2 3x3 convolutions (80 ms per position)
//   -> 17 causal LFM layers (short conv, or attention over the last 32 positions) -> a small projector into the
//   brain's 2048-wide input.
// The whole clip goes through at once, 12 positions per weight read. Weights stay as they are in the file (16-bit for
// the layers, 32-bit for the convolutions); inputs are never rounded.
#pragma once
#include "../model.h"
#include "tables.h"
#include "team.h"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

struct imi_ears {
    struct layer {
        bool attn = false;
        std::vector<float> ln1, ln2, conv_w /* [tap][E] */, q_norm, k_norm;
        imi_wg in, out;        // conv: pw1 (b | c | x), pw2; attention: q | k | v, o
        imi_wg gate_up, down;  // gate and up groups interleaved
    };
    std::vector<layer> layers;
    std::vector<float> c0_w, c0_b, c2_w, c2_b, c3_b, c5_w, c5_b, c6_b;  // downsampler; 3x3 kernels as [kh*3+kw][ch]
    imi_wg dft, c3, c6, pre, mlp1, mlp3;
    std::vector<float> hann, fb /* [n_mel][n_bin] */, pre_b, post_ln, ln_w, ln_b, mlp1_b, mlp3_b;
    int n_mel = 128, n_bin = 201, n_fft = 400, hop = 160, n_ch = 256, n_embd = 768, n_ff = 0, n_head = 0, n_head_kv = 0,
        head_dim = 0, window = 32, n_out = 0, d_conv = 3;
    float eps = 1e-5f, rope_theta = 1e6f;
    int n_threads = 4;
    double t_mel = 0, t_down = 0, t_layers = 0, t_proj = 0;   // seconds, summed over encode() calls

    bool load(const std::string & path, int threads, imi_team * shared = nullptr);
    // pcm: 16 kHz samples from the clip's start; out = [n_pos][n_out]; returns n_pos
    int encode(const float * pcm, int n, std::vector<float> & out);
    std::vector<float> last_mel;   // [n_frames][n_mel] of the last encode(), for checks
    ~imi_ears();

private:
    imi_team own_team;
    imi_team * team = &own_team;
    alignas(64) std::atomic<int> ctr{0};
    std::vector<float> xt, xres, buf, buf2;
    void par(int n_units, const std::function<void(int)> & fn);
    // out[p][row] (+)= W x_p (+ bias); xt = 12-position chunks [ch][cols][12]
    void table(const imi_wg & W, const float * xt, int P, float * out, int ldo, const float * bias, bool add);
    // gate/up group pairs -> out[p][i] = silu(gate) * up
    void table_glu(const imi_wg & W, const float * xt, int P, float * out);
    void pack(const float * a, int P, int K, int lda, float * xt);
};
