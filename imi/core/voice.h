// imi.cpp: Seoul's voice generator (LFM3-Omni depthformer) on our own engine. One audio frame = 12 codes, picked in
// masked passes; each pass runs the 12 codebook positions together through 3 small layers (pre-norm, GQA attention
// with q/k norm, NEOX rope, per-head sigmoid gate, SiLU-gated FFN; every position sees every other).
//   - the input projection depends only on the brain's hidden state: once per frame, not once per pass
//   - a pass scores only the codebooks it is about to pick
//   - 8-bit tables in our layout, multiplied with exact float inputs (12 positions per weight read)
#pragma once
#include "../model.h"
#include "tables.h"
#include "team.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

struct imi_voice {
    struct layer {
        const float *ln1, *ln2, *q_norm, *k_norm;
        imi_q8g qkvg, o, gate_up /* gate and up groups interleaved */, down;
    };
    imi_file f;
    std::vector<layer> layers;
    imi_q8g proj_in;
    std::vector<imi_q8g> score;        // per codebook: its code table, read as output scores
    const ggml_tensor * embd = nullptr, * out_embd = nullptr;  // out_embd: [n_in, n_vocab * n_cb] Q8_0, feedback rows // [n_embd, n_vocab, n_cb] Q8_0: code embeddings (input) = score tables
    const float *proj_b = nullptr, *mask_embd = nullptr, *out_norm = nullptr, *head_norm = nullptr;
    int n_embd = 0, n_vocab = 0, n_cb = 0, n_in = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0;
    float eps = 1e-5f, rope_theta = 1e6f;
    int n_threads = 4;

    std::vector<float> base;           // proj_in(h) + bias for this frame, [n_cb][n_embd]

    bool load(const std::string & path, int threads, imi_team * shared = nullptr);
    void begin_frame(const float * h);
    // the brain's next input after a frame: the sum of the 12 codes' rows of the feedback table (n_in wide)
    void feedback(const int32_t * codes, float * out) const;
    // codes[i] < 0 = still masked; writes scores for the codebooks in `need`, [need.size()][n_vocab]
    void pass(const int32_t * codes, const std::vector<int> & need, std::vector<float> & out);
    ~imi_voice();

private:
    struct per_thread { std::vector<float> x, h, xt; };
    imi_team own_team;
    imi_team * team = &own_team;
    std::vector<per_thread> per;
    std::vector<float> qkv, att, y, act, z;
    std::unique_ptr<std::atomic<int>[]> ctr;
    int n_qkv = 0;
    void run_pass(int tid, const int32_t * codes, const std::vector<int> & need, float * out);
};
