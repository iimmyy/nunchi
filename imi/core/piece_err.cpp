// imi-piece-err: research (mixed bits, step 2). For every brain table, how much each piece (8 rows x 256 columns,
// imi's reading unit) moves the table's output on the brain's real inputs, for each candidate version of the brain.
//   imi-piece-err REF.gguf REC_DIR OUT_DIR threads NAME=FILE.gguf...
// REF: the 16-bit brain. REC_DIR: hear/ tr/ voice/ from the recorder (IMI_REC; L<layer>.<slot>.f16 rows).
// For a table W, a version W_t and the 256-column block b of a piece:
//     piece error = sum over its 8 rows of  d G_b d^T,   d = (W_t - W)[row, block b],   G_b = E[x_b x_b^T]
// where the expectation weights the three jobs equally (each job's rows averaged first). It leaves out how one
// block's error lines up with the other blocks of the same row; that part is checked per table with the true output
// error on sampled input rows, E||x (W_t - W)^T||^2, next to the sum of the piece errors.
// Writes OUT_DIR/<table>.f32 (per version, in the order given: float [rows/8][cols/256]) and OUT_DIR/tables.tsv.
#include "../model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

static int n_threads = 4;
static void parallel(int n, const std::function<void(int)> & f) {
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; t++) th.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto & t : th) t.join();
}

static std::vector<float> read_rows(const std::string & path, int cols, int & n) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); const long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    n = (int) (bytes / 2 / cols);
    std::vector<ggml_fp16_t> h((size_t) n * cols);
    if (fread(h.data(), 2, h.size(), f) != h.size()) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    fclose(f);
    std::vector<float> x(h.size());
    ggml_fp16_to_fp32_row(h.data(), x.data(), (int64_t) h.size());
    return x;
}

static void to_f32(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t C = t->ne[0], R = t->ne[1];
    out.resize((size_t) R * C);
    const auto * tr = ggml_get_type_traits(t->type);
    parallel((int) R, [&](int r) {
        const char * row = (const char *) t->data + (size_t) r * t->nb[1];
        if (t->type == GGML_TYPE_F32) memcpy(&out[(size_t) r * C], row, C * sizeof(float));
        else tr->to_float(row, &out[(size_t) r * C], C);
    });
}

static double dot(const float * a, const float * b, int n) {
    double s = 0; float acc[8] = {0};
    int i = 0;
    for (; i + 8 <= n; i += 8) for (int k = 0; k < 8; k++) acc[k] += a[i + k] * b[i + k];
    for (int k = 0; k < 8; k++) s += acc[k];
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: imi-piece-err REF.gguf REC_DIR OUT_DIR threads NAME=FILE.gguf...\n"); return 1; }
    const std::string rec = argv[2], out = argv[3];
    n_threads = atoi(argv[4]);
    imi_file ref;
    if (!ref.load(argv[1])) return 1;
    std::vector<std::string> names;
    std::vector<imi_file> vers(argc - 5);
    for (int i = 5; i < argc; i++) {
        const std::string a = argv[i]; const size_t e = a.find('=');
        names.push_back(a.substr(0, e));
        if (!vers[i - 5].load(a.substr(e + 1))) return 1;
    }
    const char * jobs[3] = { "hear", "tr", "voice" };
    const int SAMPLE = 96;   // input rows per job for the true-error check
    struct tab { const char * key, * slot; };
    const tab tabs[] = { { "shortconv.in_proj", "in" }, { "attn_q", "in" }, { "attn_k", "in" }, { "attn_v", "in" },
                         { "attn_gate", "in" }, { "shortconv.out_proj", "conv_out" }, { "attn_output", "attn_out" },
                         { "ffn_gate", "ffn_in" }, { "ffn_up", "ffn_in" }, { "ffn_down", "ffn_down" } };
    FILE * tsv = fopen((out + "/tables.tsv").c_str(), "w");
    FILE * jt = fopen((out + "/jobs.tsv").c_str(), "w");   // true output error per job (hearing, translation, voice)
    fprintf(jt, "table\tversion\thear\ttr\tvoice\thear_energy\ttr_energy\tvoice_energy\n");
    fprintf(tsv, "table\trows\tcols\tslot\tenergy");
    for (auto & n : names) fprintf(tsv, "\t%s_pieces\t%s_true", n.c_str(), n.c_str());
    fprintf(tsv, "\n");
    const int n_layer = (int) ref.key_i("lfm3.block_count", 0);
    for (int il = 0; il < n_layer; il++) {
        std::string cached_slot;
        std::vector<float> G, S;   // G: [nb][256][256]; S: sampled rows [3 * SAMPLE][C]
        int nb = 0;
        for (const tab & tb : tabs) {
            const std::string name = "blk." + std::to_string(il) + "." + tb.key + ".weight";
            const ggml_tensor * t = ref.get(name, false);
            if (!t) continue;
            const int C = (int) t->ne[0], R = (int) t->ne[1];
            if (cached_slot != tb.slot) {   // second moments of this slot's inputs, 256-column blocks
                cached_slot = tb.slot;
                nb = C / 256;
                G.assign((size_t) nb * 256 * 256, 0.0f);
                S.assign((size_t) 3 * SAMPLE * C, 0.0f);
                for (int j = 0; j < 3; j++) {
                    int n = 0;
                    const std::vector<float> X = read_rows(rec + "/" + jobs[j] + "/L" + std::to_string(il) + "." + tb.slot + ".f16", C, n);
                    for (int s = 0; s < SAMPLE; s++) memcpy(&S[((size_t) j * SAMPLE + s) * C], &X[(size_t) (s * (size_t) n / SAMPLE) * C], C * sizeof(float));
                    const float w = 1.0f / (3.0f * (float) n);
                    parallel(nb * 256, [&](int bi) {   // one row i of block b's G: G[b][i][k] += w * sum_r x_ri x_rk
                        const int b = bi / 256, i = bi % 256;
                        std::vector<double> acc(256, 0.0);
                        for (int r = 0; r < n; r++) {
                            const float * xr = &X[(size_t) r * C + 256 * b];
                            const float xi = xr[i];
                            if (xi == 0.0f) continue;
                            for (int k = 0; k < 256; k++) acc[k] += (double) (xi * xr[k]);
                        }
                        float * g = &G[((size_t) b * 256 + i) * 256];
                        for (int k = 0; k < 256; k++) g[k] += w * (float) acc[k];
                    });
                }
            }
            std::vector<float> W, Q;
            to_f32(t, W);
            std::vector<double> tr_err((size_t) R), en((size_t) R), tj((size_t) R * 3), ej((size_t) R * 3);
            parallel(R, [&](int r) {   // the table's output energy on the sampled rows
                double s = 0;
                for (int j = 0; j < 3 * SAMPLE; j++) { const double y = dot(&S[(size_t) j * C], &W[(size_t) r * C], C); s += y * y; ej[(size_t) r * 3 + j / SAMPLE] += y * y / SAMPLE; }
                en[r] = s / (3 * SAMPLE);
            });
            double EJ[3] = {0, 0, 0};
            for (int r = 0; r < R; r++) for (int k = 0; k < 3; k++) EJ[k] += ej[(size_t) r * 3 + k];
            double energy = 0; for (double v : en) energy += v;
            fprintf(tsv, "%s\t%d\t%d\t%s\t%.6e", name.c_str(), R, C, tb.slot, energy);
            FILE * bf = fopen((out + "/" + name + ".f32").c_str(), "wb");
            printf("%s %d x %d:", name.c_str(), R, C);
            for (size_t v = 0; v < vers.size(); v++) {
                const ggml_tensor * tq = vers[v].get(name);
                to_f32(tq, Q);
                std::vector<double> eb((size_t) R * nb);   // error of each (row, block)
                parallel(R, [&](int r) {
                    std::vector<float> dr(C);
                    for (int k = 0; k < C; k++) dr[k] = Q[(size_t) r * C + k] - W[(size_t) r * C + k];
                    for (int b = 0; b < nb; b++) {   // d G_b d^T
                        const float * d = &dr[256 * b], * g = &G[(size_t) b * 256 * 256];
                        double e = 0;
                        for (int i = 0; i < 256; i++) if (d[i] != 0.0f) e += (double) d[i] * dot(g + (size_t) i * 256, d, 256);
                        eb[(size_t) r * nb + b] = e;
                    }
                    double s = 0;   // true output error on the sampled rows
                    for (int k = 0; k < 3; k++) tj[(size_t) r * 3 + k] = 0;
                    for (int j = 0; j < 3 * SAMPLE; j++) { const double y = dot(&S[(size_t) j * C], dr.data(), C); s += y * y; tj[(size_t) r * 3 + j / SAMPLE] += y * y / SAMPLE; }
                    tr_err[r] = s / (3 * SAMPLE);
                });
                std::vector<float> pe((size_t) (R / 8) * nb, 0.0f);   // pieces: 8 rows of one block
                double psum = 0, tsum = 0;
                for (int r = 0; r < R; r++) {
                    for (int b = 0; b < nb; b++) { pe[(size_t) (r / 8) * nb + b] += (float) eb[(size_t) r * nb + b]; psum += eb[(size_t) r * nb + b]; }
                    tsum += tr_err[r];
                }
                fprintf(tsv, "\t%.6e\t%.6e", psum, tsum);
                double TJ[3] = {0, 0, 0};
                for (int r = 0; r < R; r++) for (int k = 0; k < 3; k++) TJ[k] += tj[(size_t) r * 3 + k];
                fprintf(jt, "%s\t%s\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n", name.c_str(), names[v].c_str(), TJ[0], TJ[1], TJ[2], EJ[0], EJ[1], EJ[2]);
                fflush(jt);
                printf("  %s %.2e (pieces/true %.2f)", names[v].c_str(), tsum / energy, psum / std::max(tsum, 1e-30));
                fwrite(pe.data(), sizeof(float), pe.size(), bf);
            }
            printf("\n"); fflush(stdout);
            fclose(bf);
            fprintf(tsv, "\n"); fflush(tsv);
        }
    }
    fclose(tsv); fclose(jt);
    printf("PIECES-DONE\n");
    return 0;
}
