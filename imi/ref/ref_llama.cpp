// The answer key for imi's brain: llama.cpp runs a text one position at a time and saves, per position, the word,
// the hidden state (after the last norm) and the next-word scores. Build against our llama.cpp fork:
//   c++ -O2 -std=c++17 ref_llama.cpp -I<fork>/include -I<fork>/ggml/include -L<fork>/build/bin -lllama -lggml -o ref_llama
//   ref_llama MODEL.gguf TEXT.txt OUT.bin [threads] [batch]   (batch = the whole text in one decode: llama.cpp's own other path)
// record: int32 n_embd, int32 n_vocab, int32 n_tok; per position: int32 token, f32 hidden[n_embd], f32 logits[n_vocab]
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: ref_llama MODEL.gguf TEXT.txt OUT.bin [threads]\n"); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    const bool batch = argc > 5 && std::string(argv[5]) == "batch";
    cp.n_ctx = 1024; cp.n_batch = batch ? 1024 : 1; cp.n_ubatch = batch ? 1024 : 1;
    cp.embeddings = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads = cp.n_threads_batch = argc > 4 ? atoi(argv[4]) : 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::stringstream ss; ss << std::ifstream(argv[2]).rdbuf();
    const std::string text = ss.str();
    std::vector<llama_token> toks(text.size() + 8);
    const int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
    toks.resize(n);

    const int32_t ne = llama_model_n_embd(model), nv = llama_vocab_n_tokens(vocab);
    FILE * f = fopen(argv[3], "wb");
    fwrite(&ne, 4, 1, f); fwrite(&nv, 4, 1, f); fwrite(&n, 4, 1, f);
    if (batch) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) { b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int i = 0; i < n; i++) {
            fwrite(&toks[i], 4, 1, f);
            fwrite(llama_get_embeddings_ith(ctx, i), 4, ne, f);
            fwrite(llama_get_logits_ith(ctx, i), 4, nv, f);
        }
    }
    for (int i = 0; i < n && !batch; i++) {
        llama_batch b = llama_batch_get_one(&toks[i], 1);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
        fwrite(&toks[i], 4, 1, f);
        fwrite(llama_get_embeddings_ith(ctx, -1), 4, ne, f);
        fwrite(llama_get_logits_ith(ctx, -1), 4, nv, f);
    }
    fclose(f);
    fprintf(stderr, "ref_llama: %d positions\n", n);
    return 0;
}
