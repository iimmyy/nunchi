// imi.cpp: Seoul's brain (LFM3) on our own engine, written from the model's math. No graph, no planner: one pass
// over the layers, our own thread team, and a stop only where the next piece of math needs every thread's rows.
//
// Per layer: norm -> (short conv | gated attention) -> norm -> residual, then norm -> SiLU-gated FFN -> norm -> residual.
// Stops per layer: conv 4 (in_proj, out_proj, ffn gate|up, ffn down), attention 5 (q|k|v|gate, attention, output,
// ffn gate|up, ffn down). Every thread keeps its own copy of the running state and does the small math (norms, adds,
// conv, input cutting) itself, which is cheaper than waiting for one thread to do it.
#pragma once
#include "../model.h"
#include "tables.h"
#include "team.h"

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

struct imi_seoul {
    struct layer {
        const float *attn_norm, *post_attn_norm, *ffn_norm, *post_ffn_norm;
        const float *conv_w = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        imi_q4 in_proj, out_proj, qkvg, o, gate_up /* gate and up groups interleaved */, down;
        bool conv = false, swa = false;
        int ai = -1;
    };

    imi_file f;
    std::vector<layer> layers;
    const uint8_t * tok = nullptr;     // word table, Q6_K rows (input lookup and word scores)
    size_t tok_row = 0;
    const float * out_norm = nullptr, * rope_freqs = nullptr;
    int n_embd = 0, n_vocab = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0, d_conv = 0, n_swa = 0, n_ctx = 0;
    float eps = 1e-5f, rope_base = 1e6f, rope_base_swa = 1e4f;

    // memory carried between steps
    std::vector<std::vector<uint16_t>> k_cache, v_cache;   // per attention layer: [n_ctx][kv_dim] f16
    std::vector<std::vector<float>> conv_state;            // per layer: [d_conv][n_embd], oldest first
    std::deque<std::vector<float>> conv_w_t;               // conv weights as [tap][channel]
    int pos = 0;

    std::vector<float> hidden, logits;

    int n_threads = 4;
    double t_tables = 0, t_sync = 0, t_attn = 0;   // research: thread 0's time in table reads, waiting at stops, attention (ms)

    // shared: a thread team to run on (the voice generator uses the same one); nullptr = start our own
    bool load(const std::string & path, int threads, int ctx_len, imi_team * shared = nullptr);
    void reset();
    void step(int32_t token, const float * embd, bool want_logits);
    // read many positions at once (a prompt, or the ears' audio vectors): tokens, or embd = [n][n_embd].
    // Every table is read once per 64 positions. hidden (and logits, if asked) are the last position's;
    // hidden_all, if given, gets every position's hidden state ([n][n_embd]).
    void step_batch(const int32_t * tokens, const float * embd, int n, bool want_logits, float * hidden_all = nullptr);
    // research: every layer table read back to back, ncol (1 or 2) inputs riding each read; best ms over reps
    double bench_tables(int ncol, int reps);
    ~imi_seoul();

private:
    struct per_thread {
        std::vector<float> x, h, t, u;   // running state, norm out, scratch, conv out
        imi_q8 qi; imi_q8k qk;
    };
    imi_team own_team;
    imi_team * team = &own_team;
    std::vector<per_thread> per;
    std::vector<float> bcx, qkv, att, y, act, fo;   // shared: written by row slices, read after a stop
    std::unique_ptr<std::atomic<int>[]> ctr;
    int n_ctr = 512;
    void run(int tid, int32_t token, const float * embd, bool want_logits);
    // batch reading
    static constexpr int NB = 64;
    std::vector<float> bX, bH, bBig, bY, bAct;   // [NB][...]
    imi_q8b bq;
    void run_batch(int tid, const int32_t * tokens, const float * embd, int n, float * hidden_all);
};
