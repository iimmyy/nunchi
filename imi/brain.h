// imi.cpp: Seoul's brain (LFM3), one position per step, written from the model's math.
//
// 24 layers, each: norm -> (short conv | gated attention) -> norm -> residual, then norm -> SiLU-gated FFN -> norm ->
// residual. Short conv: in_proj splits into b, c, x; b*x goes through a 7-tap causal depthwise conv over the last 6
// steps (our conv memory), times c, then out_proj. Attention: 16 query heads share 4 key/value heads (size 128), q/k
// RMS-normed, NEOX rope (base 1e6 with the frequency table on full layers, 1e4 on the 256-step window layers), a
// per-head sigmoid gate read from the layer input, then the output table. The last norm gives the hidden state the
// voice reads; the tied word table on top gives next-word scores.
#pragma once
#include "model.h"
#include "ggml-cpu.h"
#include "kernels/q4x8.h"

#include <unordered_map>

#include <cstdint>
#include <string>
#include <vector>

struct imi_brain {
    struct layer {
        ggml_tensor *attn_norm, *post_attn_norm, *ffn_norm, *post_ffn_norm, *ffn_up, *ffn_gate, *ffn_down;
        // short conv
        ggml_tensor *conv_w = nullptr, *in_proj = nullptr, *out_proj = nullptr;
        // attention
        ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *gate = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        bool swa = false;
        // stacked tables (fewer stops): q|k|v|gate and ffn gate|up, in the fast-kernel layout
        ggml_tensor *qkvg = nullptr, *gate_up = nullptr;
        int  attn_index = -1;
    };

    imi_file f;
    ggml_tensor *tok_embd, *out_norm, *rope_freqs;
    std::vector<layer> layers;
    int n_embd = 0, n_vocab = 0, n_head = 0, n_head_kv = 0, head_dim = 0, d_conv = 0, n_swa = 0, n_ctx = 0;
    float eps = 1e-5f, rope_base = 1e6f, rope_base_swa = 1e4f;

    // memory carried between steps
    ggml_context * mem_ctx = nullptr;
    ggml_context * fuse_ctx = nullptr;          // headers of the stacked tables
    ggml_backend_buffer_t fuse_buf = nullptr;   // their bytes
    std::vector<ggml_tensor *> k_cache, v_cache;  // per attention layer: K [kv_dim, n_ctx] f16, V transposed [n_ctx, kv_dim] f16
    std::vector<ggml_tensor *> conv_state;        // per conv layer: [d_conv, n_embd] f32
    int pos = 0;

    int n_threads = 4;
    ggml_threadpool * tp = nullptr;
    std::vector<uint8_t> arena, work;

    std::vector<float> hidden, logits;

    // our kernel's copies of the 4-bit tables (IMI_KERNEL=ggml uses ggml's instead)
    std::unordered_map<const ggml_tensor *, imi_q4x8 *> ours;
    std::vector<imi_q4x8_call> calls;  // this step's uses of our tables (counters reset every step)
    size_t n_calls = 0;
    int last_nodes = 0, last_compute_nodes = 0;  // graph size of the last step (stops = compute nodes)
    ggml_tensor * mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x);

    bool load(const std::string & path, int threads, int ctx_len);
    void reset();
    // one step: a word (token >= 0) or a ready input vector (embd, e.g. the voice's feedback). want_logits = score words
    void step(int32_t token, const float * embd, bool want_logits);
    ~imi_brain();
};
