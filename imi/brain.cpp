#include "brain.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>

bool imi_brain::load(const std::string & path, int threads, int ctx_len) {
    const char * kern = getenv("IMI_KERNEL");                       // "ggml" = ggml's fast layout, default = ours
    const bool use_ours = kern && strcmp(kern, "ours") == 0;           // our q4x8 kernel (parked: loses in place)
    const bool stack = !use_ours && getenv("IMI_NO_STACK") == nullptr;  // default: stacked tables on ggml's fast kernel
    if (!f.load(path, /*repack*/ !use_ours && !stack && getenv("IMI_NO_REPACK") == nullptr)) return false;
    tok_embd   = f.get("token_embd.weight");
    out_norm   = f.get("output_norm.weight");
    rope_freqs = f.get("rope_freqs.weight", false);
    n_embd   = (int) tok_embd->ne[0];
    n_vocab  = (int) tok_embd->ne[1];
    eps           = f.key_f("lfm3.attention.layer_norm_rms_epsilon", 1e-5f);
    rope_base     = f.key_f("lfm3.rope.freq_base", 1e6f);
    rope_base_swa = f.key_f("lfm3.rope.freq_base_swa", 1e4f);
    n_swa         = (int) f.key_i("lfm3.attention.sliding_window", 0);
    d_conv        = (int) f.key_i("lfm3.shortconv.l_cache", 7) - 1;
    n_head        = (int) f.key_i("lfm3.attention.head_count", 16);
    head_dim      = (int) f.key_i("lfm3.attention.key_length", 128);
    const int n_layer = (int) f.key_i("lfm3.block_count", 0);

    // which attention layers use the short window (an array of bools in the file)
    std::vector<bool> swa(n_layer, false);
    const int64_t kswa = gguf_find_key(f.meta, "lfm3.attention.sliding_window_pattern");
    if (kswa >= 0) {
        const bool * a = (const bool *) gguf_get_arr_data(f.meta, kswa);
        for (int i = 0; i < n_layer && i < (int) gguf_get_arr_n(f.meta, kswa); i++) swa[i] = a[i];
    }

    int n_attn = 0;
    for (int il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        layer L;
        L.attn_norm      = f.get(p + "attn_norm.weight");
        L.post_attn_norm = f.get(p + "post_attention_norm.weight");
        L.ffn_norm       = f.get(p + "ffn_norm.weight");
        L.post_ffn_norm  = f.get(p + "post_ffw_norm.weight");
        L.ffn_up   = f.get(p + "ffn_up.weight");
        L.ffn_gate = f.get(p + "ffn_gate.weight");
        L.ffn_down = f.get(p + "ffn_down.weight");
        if (f.get(p + "shortconv.conv.weight", false)) {
            L.conv_w   = f.get(p + "shortconv.conv.weight");
            L.in_proj  = f.get(p + "shortconv.in_proj.weight");
            L.out_proj = f.get(p + "shortconv.out_proj.weight");
        } else {
            L.q = f.get(p + "attn_q.weight");  L.k = f.get(p + "attn_k.weight");
            L.v = f.get(p + "attn_v.weight");  L.o = f.get(p + "attn_output.weight");
            L.gate   = f.get(p + "attn_gate.weight");
            L.q_norm = f.get(p + "attn_q_norm.weight");
            L.k_norm = f.get(p + "attn_k_norm.weight");
            L.swa = swa[il];
            L.attn_index = n_attn++;
            n_head_kv = (int) (L.k->ne[1] / head_dim);
        }
        layers.push_back(L);
    }

    // conv memory and attention memory
    n_ctx = ctx_len;
    const int kv_dim = n_head_kv * head_dim;
    const size_t mem = (size_t) n_attn * 2 * ggml_row_size(GGML_TYPE_F16, (int64_t) kv_dim * n_ctx)
                     + (size_t) n_layer * d_conv * n_embd * sizeof(float) + (size_t) 4 * n_layer * ggml_tensor_overhead();
    ggml_init_params ip = { mem, nullptr, false };
    mem_ctx = ggml_init(ip);
    for (const auto & L : layers) {
        if (L.conv_w) {
            conv_state.push_back(ggml_new_tensor_2d(mem_ctx, GGML_TYPE_F32, d_conv, n_embd));
        } else {
            conv_state.push_back(nullptr);
            k_cache.push_back(ggml_new_tensor_2d(mem_ctx, GGML_TYPE_F16, kv_dim, n_ctx));
            v_cache.push_back(ggml_new_tensor_2d(mem_ctx, GGML_TYPE_F16, n_ctx, kv_dim));
        }
    }
    reset();

    if (stack) {
        // stack the tables that read the same input into one table: Q4_0 rows are independent, so stacking is placing
        // the bytes end to end. Every 4-bit layer table then goes into the CPU's fast-kernel layout.
        ggml_backend_buffer_type_t rb = imi_repack_buft();
        std::vector<std::pair<ggml_tensor *, std::vector<ggml_tensor *>>> plan;
        ggml_init_params fp = { 4096 * ggml_tensor_overhead(), nullptr, true };
        fuse_ctx = ggml_init(fp);
        auto stacked = [&](std::vector<ggml_tensor *> parts, const char * name, int il) -> ggml_tensor * {
            int64_t rows = 0;
            for (auto * p : parts) { if (p->type != parts[0]->type || p->ne[0] != parts[0]->ne[0]) return nullptr; rows += p->ne[1]; }
            ggml_tensor * t = ggml_new_tensor_2d(fuse_ctx, parts[0]->type, parts[0]->ne[0], rows);
            ggml_format_name(t, "blk.%d.%s", il, name);
            plan.push_back({ t, parts });
            return t;
        };
        size_t need = 0;
        for (size_t il = 0; il < layers.size(); il++) {
            layer & L = layers[il];
            // the layer's own fields are repointed, so the step reads the fast-layout copies
            std::vector<ggml_tensor **> singles;
            if (L.conv_w) singles = { &L.in_proj, &L.out_proj }; else { singles = { &L.o }; L.qkvg = stacked({ L.q, L.k, L.v, L.gate }, "qkvg", (int) il); }
            L.gate_up = stacked({ L.ffn_gate, L.ffn_up }, "gate_up", (int) il);
            singles.push_back(&L.ffn_down);
            for (ggml_tensor ** s : singles) {
                if ((*s)->type != GGML_TYPE_Q4_0) continue;            // Q4_1 tables stay as stored
                *s = stacked({ *s }, "fast", (int) il);
            }
        }
        for (auto & pr : plan) need += GGML_PAD(ggml_backend_buft_get_alloc_size(rb, pr.first), 64);
        fuse_buf = ggml_backend_buft_alloc_buffer(rb, need + 4096);
        ggml_backend_buffer_set_usage(fuse_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tallocr ta = ggml_tallocr_new(fuse_buf);
        std::vector<uint8_t> bytes;
        for (auto & pr : plan) {
            bytes.clear();
            for (auto * p : pr.second) bytes.insert(bytes.end(), (const uint8_t *) p->data, (const uint8_t *) p->data + ggml_nbytes(p));
            ggml_tallocr_alloc(&ta, pr.first);
            ggml_backend_tensor_set(pr.first, bytes.data(), 0, bytes.size());
        }
        fprintf(stderr, "imi-brain: %zu tables in the fast-kernel layout, same-input tables stacked (%zu MB)\n", plan.size(), need >> 20);
    }
    if (use_ours) {
        size_t nb = 0;
        for (auto & L : layers) {
            for (ggml_tensor * w : { L.in_proj, L.out_proj, L.q, L.k, L.v, L.o, L.gate, L.ffn_up, L.ffn_gate, L.ffn_down }) {
                if (w && imi_q4x8_supported(w)) { ours[w] = imi_q4x8_from(w); nb += ggml_nbytes(w); }
            }
        }
        calls = std::vector<imi_q4x8_call>(ours.size() + 8);
        fprintf(stderr, "imi-brain: %zu tables on our kernel (%zu MB)\n", ours.size(), nb >> 20);
    }
    n_threads = threads;
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    tp = ggml_threadpool_new(&tpp);
    arena.resize(256u << 20);
    fprintf(stderr, "imi-brain: %d layers (%d attention, %d heads / %d kv x %d, window %d), %d words, conv %d taps\n",
            n_layer, n_attn, n_head, n_head_kv, head_dim, n_swa, n_vocab, d_conv + 1);
    return true;
}

void imi_brain::reset() {
    pos = 0;
    for (auto * t : conv_state) if (t) memset(t->data, 0, ggml_nbytes(t));
}

void imi_brain::step(int32_t token, const float * embd, bool want_logits) {
    ggml_init_params ip = { arena.size(), arena.data(), false };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    const int E = n_embd, D = head_dim, kv_dim = n_head_kv * head_dim;

    std::vector<ggml_tensor *> dbg;
    n_calls = 0;
    ggml_tensor * x;
    if (token >= 0) {
        ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *) id->data)[0] = token;
        x = ggml_get_rows(ctx, tok_embd, id);
    } else {
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, 1);
        memcpy(x->data, embd, (size_t) E * sizeof(float));
    }
    auto tag = [&](ggml_tensor * t, const char * nm, size_t il) { if (il == 0) { ggml_format_name(t, "%s-0", nm); dbg.push_back(t); } return t; };
    tag(x, "embed", 0);
    ggml_tensor * ppos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ((int32_t *) ppos->data)[0] = pos;

    for (size_t il = 0; il < layers.size(); il++) {
        const layer & L = layers[il];
        ggml_tensor * h = tag(ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), L.attn_norm), "operator_norm", il);
        ggml_tensor * y;
        if (L.conv_w) {
            ggml_tensor * bcx = tag(mm(ctx, L.in_proj, h), "in_proj", il);     // [3E]
            ggml_tensor * b  = ggml_view_1d(ctx, bcx, E, 0);
            ggml_tensor * c  = ggml_view_1d(ctx, bcx, E, (size_t) E * sizeof(float));
            ggml_tensor * xx = ggml_view_1d(ctx, bcx, E, (size_t) 2 * E * sizeof(float));
            ggml_tensor * bx = ggml_reshape_2d(ctx, ggml_mul(ctx, b, xx), 1, E);          // [1 step, E]
            ggml_tensor * st = conv_state[il];
            ggml_tensor * sx = ggml_concat(ctx, st, bx, 0);                                // [d_conv + 1, E]
            ggml_tensor * co = (ggml_tensor *) nullptr; co = ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, sx, d_conv + 1, E, 1), L.conv_w); // [E, 1, 1]
            // the conv memory slides one step: the last d_conv inputs
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_2d(ctx, sx, d_conv, E, sx->nb[1], sx->nb[0]), st));
            tag(co, "conv", il);
            y = tag(mm(ctx, L.out_proj, ggml_mul(ctx, c, ggml_reshape_1d(ctx, co, E))), "out_proj", il);
        } else {
            const int ia = L.attn_index;
            ggml_tensor *q, *k, *v, *graw;
            if (L.qkvg) {                                                      // one read, one stop for all four
                ggml_tensor * all = ggml_mul_mat(ctx, L.qkvg, h);
                const size_t fs = sizeof(float);
                q    = ggml_view_1d(ctx, all, (int64_t) D * n_head, 0);
                k    = ggml_view_1d(ctx, all, kv_dim, (size_t) D * n_head * fs);
                v    = ggml_view_1d(ctx, all, kv_dim, (size_t) (D * n_head + kv_dim) * fs);
                graw = ggml_view_1d(ctx, all, n_head, (size_t) (D * n_head + 2 * kv_dim) * fs);
            } else {
                q = mm(ctx, L.q, h); k = mm(ctx, L.k, h); v = mm(ctx, L.v, h); graw = mm(ctx, L.gate, h);
            }
            q = ggml_reshape_3d(ctx, q, D, n_head, 1);
            k = ggml_reshape_3d(ctx, k, D, n_head_kv, 1);
            q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), L.q_norm);
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), L.k_norm);
            const float base = L.swa ? rope_base_swa : rope_base;
            ggml_tensor * ff = L.swa ? nullptr : rope_freqs;
            q = ggml_rope_ext(ctx, q, ppos, ff, D, GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k = ggml_rope_ext(ctx, k, ppos, ff, D, GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

            // write this step into the attention memory (K by position, V transposed), then read the visible span
            ggml_tensor * kc = k_cache[ia], * vc = v_cache[ia];
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, k, kv_dim),
                ggml_view_1d(ctx, kc, kv_dim, (size_t) pos * kc->nb[1])));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, v, 1, kv_dim),
                ggml_view_2d(ctx, vc, 1, kv_dim, vc->nb[1], (size_t) pos * vc->nb[0])));
            const int s = (L.swa && n_swa > 0) ? std::max(0, pos - n_swa + 1) : 0;
            const int n_kv = pos - s + 1;
            ggml_tensor * K = ggml_view_3d(ctx, kc, D, n_head_kv, n_kv, (size_t) D * kc->nb[0], kc->nb[1], (size_t) s * kc->nb[1]);
            ggml_tensor * V = ggml_view_3d(ctx, vc, n_kv, D, n_head_kv, vc->nb[1], (size_t) D * vc->nb[1], (size_t) s * vc->nb[0]);
            K = ggml_permute(ctx, K, 0, 2, 1, 3);                                          // [D, n_kv, n_head_kv]
            ggml_tensor * Q = ggml_permute(ctx, q, 0, 2, 1, 3);                            // [D, 1, n_head]
            ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), nullptr, 1.0f / sqrtf((float) D), 0.0f);
            ggml_tensor * a = ggml_mul_mat(ctx, V, kq);                                    // [D, 1, n_head]
            a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));                          // [D, n_head, 1]
            ggml_tensor * g = ggml_sigmoid(ctx, graw);                          // [n_head]
            a = ggml_mul(ctx, a, ggml_reshape_3d(ctx, g, 1, n_head, 1));
            y = mm(ctx, L.o, ggml_reshape_2d(ctx, a, D * n_head, 1));
        }
        y = tag(ggml_mul(ctx, ggml_rms_norm(ctx, y, eps), L.post_attn_norm), "post_operator_norm", il);
        x = tag(ggml_add(ctx, x, y), "mid", il);

        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), L.ffn_norm);
        ggml_tensor *fg, *fu;
        if (L.gate_up) {
            ggml_tensor * gu = ggml_mul_mat(ctx, L.gate_up, h);
            const int64_t F = L.ffn_gate->ne[1];
            fg = ggml_view_1d(ctx, gu, F, 0);
            fu = ggml_view_1d(ctx, gu, F, (size_t) F * sizeof(float));
        } else {
            fg = mm(ctx, L.ffn_gate, h); fu = mm(ctx, L.ffn_up, h);
        }
        ggml_tensor * fo = mm(ctx, L.ffn_down, ggml_mul(ctx, ggml_silu(ctx, fg), fu));
        tag(fo, "ffn_out", il);
        fo = tag(ggml_mul(ctx, ggml_rms_norm(ctx, fo, eps), L.post_ffn_norm), "post_ffn_norm", il);
        x = ggml_add(ctx, x, fo);
        ggml_format_name(x, "l_out-%zu", il);
        dbg.push_back(x);
    }
    ggml_tensor * hid = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), out_norm);
    ggml_build_forward_expand(gf, hid);
    ggml_tensor * lg = nullptr;
    if (want_logits) {
        lg = ggml_mul_mat(ctx, tok_embd, hid);
        ggml_build_forward_expand(gf, lg);
    }

    last_nodes = ggml_graph_n_nodes(gf); last_compute_nodes = 0;
    for (int i = 0; i < last_nodes; i++) {
        const ggml_op op = ggml_graph_node(gf, i)->op;
        if (op != GGML_OP_NONE && op != GGML_OP_VIEW && op != GGML_OP_RESHAPE && op != GGML_OP_PERMUTE && op != GGML_OP_TRANSPOSE) last_compute_nodes++;
    }
    ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    if (work.size() < plan.work_size) work.resize(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    if (getenv("IMI_DEBUG") && pos == 0) {
        for (ggml_tensor * t : dbg) {
            double sum = 0; const float * d = (const float *) t->data;
            for (int64_t i = 0; i < ggml_nelements(t); i++) sum += d[i];
            fprintf(stderr, "%-22s sum = %12.6f  first %.4f %.4f %.4f\n", t->name, sum, d[0], d[1], d[2]);
        }
    }
    hidden.assign((float *) hid->data, (float *) hid->data + E);
    if (lg) logits.assign((float *) lg->data, (float *) lg->data + n_vocab);
    ggml_free(ctx);
    pos++;
}

ggml_tensor * imi_brain::mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    auto it = ours.find(w);
    if (it != ours.end() && x->ne[1] == 1 && n_calls < calls.size()) {
        imi_q4x8_call & c = calls[n_calls++];
        c.w = it->second; c.next.store(0);
        return ggml_reshape_2d(ctx, imi_q4x8_mul(ctx, &c, ggml_reshape_1d(ctx, x, x->ne[0])), w->ne[1], 1);
    }
    return ggml_mul_mat(ctx, w, x);
}

imi_brain::~imi_brain() {
    if (fuse_buf) ggml_backend_buffer_free(fuse_buf);
    if (fuse_ctx) ggml_free(fuse_ctx);
    for (auto & kv : ours) delete kv.second;
    if (mem_ctx) ggml_free(mem_ctx);
    if (tp) ggml_threadpool_free(tp);
}
