// imi-hear-pieces: research, hearing short pieces (cut at the speaker's pauses) one at a time with the normal hearing
// prompt (the same as imi-hear), in one process. Writes each piece's Korean next to it (PIECE.ko.txt) and prints
// one line per piece: sound seconds, words, ms. Pieces named wN-KK.f32: the earlier pieces are the same wN's.
// IMI_CTX_S=S (research): each piece is heard with the earlier pieces (whole pieces, up to S seconds before it) in front
// of it, and the Korean already written for them goes in as the start of the answer, so the model writes only the new
// piece's words. The layout is the trained one (one clip, one answer). All of it is heard again for every piece.
// IMI_KEEP_S=S (research): the same layout, but the earlier audio is heard once. The memory keeps the run's audio so far
// (up to S seconds) and after each answer goes back to the end of that audio (the attention caches are written by
// position, so going back is setting the position and the conv memory back). The next piece's audio goes in there,
// then the answer starts again with the Korean already written. Past S seconds it starts over from the last
// IMI_KEEP_O (default 4) seconds of pieces. IMI_KEEP_CHECK=1 also hears every piece the long way, in a second memory,
// and prints how far the two answers' word scores are apart (truth: 0).
// IMI_TURNS=1 (research): a chat. Each piece is one more user turn in the same memory; nothing is given back.
// In both keeping modes each piece's ears run on the piece alone.
//   imi-hear-pieces MODEL.gguf MMPROJ.gguf threads P1.f32 [P2.f32 ...]
#include "ears.h"
#include "seoul.h"
#include "words.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: imi-hear-pieces MODEL.gguf MMPROJ.gguf threads P1.f32 [P2.f32 ...]\n"); return 1; }
    const int threads = atoi(argv[3]);
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    imi_team team;
    team.start(threads);
    imi_seoul brain;
    if (!brain.load(argv[1], threads, 4096, &team)) return 1;
    imi_words words;
    if (!words.load(brain.f)) return 1;
    imi_ears ears;
    if (!ears.load(argv[2], threads, &team)) return 1;
    const std::vector<int32_t> pre = words.encode_prompt("<|startoftext|><|im_start|>system\nPerform ASR in Korean.<|im_end|>\n<|im_start|>user\n<|vision_start|>"),
                               post = words.encode_prompt("<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>");
    const double ctx_s = getenv("IMI_CTX_S") ? atof(getenv("IMI_CTX_S")) : 0.0;
    const double keep_s = getenv("IMI_KEEP_S") ? atof(getenv("IMI_KEEP_S")) : 0.0;
    const double keep_o = getenv("IMI_KEEP_O") ? atof(getenv("IMI_KEEP_O")) : 4.0;
    const bool keep_check = getenv("IMI_KEEP_CHECK") && atoi(getenv("IMI_KEEP_CHECK")) > 0;
    const bool turns = getenv("IMI_TURNS") && atoi(getenv("IMI_TURNS")) > 0;
    const std::vector<int32_t> turn = words.encode_prompt("<|im_end|>\n<|im_start|>user\n<|vision_start|>");
    struct earlier { std::vector<float> pcm, emb; int P = 0; std::vector<int32_t> ids; };
    std::vector<earlier> before;   // the earlier pieces of this run (IMI_KEEP_S: of this stretch of memory), oldest first
    std::string run_name;
    bool open = false;             // IMI_KEEP_S / IMI_TURNS: the memory holds this run so far
    int snap_pos = 0;              // IMI_KEEP_S: the end of the audio so far
    std::vector<std::vector<float>> snap_conv;
    double kept_s = 0;
    int restarts = 0;
    double check_worst = 0;
    imi_seoul_state second;        // IMI_KEEP_CHECK: the long way's memory
    if (keep_check) second = brain.new_state();
    double sound_all = 0, t_all = 0, t_ears = 0, t_read = 0, t_gen = 0;   // research: where the time goes
    int n_prefix_all = 0, n_gen_all = 0, n_read_all = 0;
    auto trimmed = [&](std::vector<int32_t> ids) {   // given back without a trailing line break or space
        while (!ids.empty()) {
            const std::string t = words.text(&ids.back(), 1);
            if (t.find_first_not_of(" \n") != std::string::npos) break;
            ids.pop_back();
        }
        return ids;
    };
    for (int a = 4; a < argc; a++) {
        const std::string path = argv[a];
        std::vector<float> pcm;
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "no clip %s\n", path.c_str()); return 1; }
        float s; while (fread(&s, 4, 1, f) == 1) pcm.push_back(s);
        fclose(f);
        const double snd = pcm.size() / 16000.0;
        const size_t sl = path.find_last_of('/'), dash = path.find_last_of('-');
        const std::string name = path.substr(sl == std::string::npos ? 0 : sl + 1, dash - (sl == std::string::npos ? 0 : sl + 1));
        if (name != run_name) { before.clear(); run_name = name; open = false; }
        std::vector<int32_t> prefix;
        double have = 0;
        int n_read = 0, P = 0;
        double te = 0;
        const auto t0 = clk::now();
        std::vector<float> emb;
        if (keep_s > 0) {
            P = ears.encode(pcm.data(), (int) pcm.size(), emb);
            te = secs(t0);
            if (open && kept_s + snd > keep_s) {   // start over from the last few seconds of pieces
                size_t first = before.size();
                double o = 0;
                while (first > 0 && o + before[first - 1].pcm.size() / 16000.0 <= keep_o) { first--; o += before[first].pcm.size() / 16000.0; }
                before.erase(before.begin(), before.begin() + first);
                open = false;
                restarts++;
            }
            if (!open) {
                brain.reset();
                brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
                n_read += (int) pre.size();
                kept_s = 0;
                for (auto & e : before) { brain.step_batch(nullptr, e.emb.data(), e.P, false); n_read += e.P; kept_s += e.pcm.size() / 16000.0; }
                open = true;
            } else {
                brain.pos = snap_pos;
                brain.conv_state = snap_conv;
            }
            brain.step_batch(nullptr, emb.data(), P, false);
            n_read += P;
            kept_s += snd;
            snap_pos = brain.pos;
            snap_conv = brain.conv_state;
            for (auto & e : before) { prefix.insert(prefix.end(), e.ids.begin(), e.ids.end()); have += e.pcm.size() / 16000.0; }
            brain.step_batch(post.data(), nullptr, (int) post.size(), prefix.empty());
            if (!prefix.empty()) brain.step_batch(prefix.data(), nullptr, (int) prefix.size(), true);
            n_read += (int) (post.size() + prefix.size());
            if (keep_check) {   // the long way: everything read from the start in a second memory
                brain.swap_state(second);
                brain.reset();
                brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
                for (auto & e : before) brain.step_batch(nullptr, e.emb.data(), e.P, false);
                brain.step_batch(nullptr, emb.data(), P, false);
                brain.step_batch(post.data(), nullptr, (int) post.size(), prefix.empty());
                if (!prefix.empty()) brain.step_batch(prefix.data(), nullptr, (int) prefix.size(), true);
                std::vector<float> l2 = brain.logits;
                brain.swap_state(second);
                double d = 0;
                for (int v = 0; v < brain.n_vocab; v++) d = std::max(d, (double) std::fabs(brain.logits[v] - l2[v]));
                check_worst = std::max(check_worst, d);
            }
        } else if (turns) {
            P = ears.encode(pcm.data(), (int) pcm.size(), emb);
            te = secs(t0);
            if (open && brain.pos + P + 600 > 4096) { open = false; restarts++; }
            if (!open) {
                brain.reset();
                brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
                n_read += (int) pre.size();
                open = true;
            } else {
                brain.step_batch(turn.data(), nullptr, (int) turn.size(), false);   // closes the last answer, opens a turn
                n_read += (int) turn.size();
            }
            brain.step_batch(nullptr, emb.data(), P, false);
            brain.step_batch(post.data(), nullptr, (int) post.size(), true);
            n_read += P + (int) post.size();
        } else {
            // the earlier pieces that fit in ctx_s seconds, newest last
            size_t first = before.size();
            while (first > 0 && have + before[first - 1].pcm.size() / 16000.0 <= ctx_s) { first--; have += before[first].pcm.size() / 16000.0; }
            std::vector<float> clip;
            for (size_t i = first; i < before.size(); i++) {
                clip.insert(clip.end(), before[i].pcm.begin(), before[i].pcm.end());
                prefix.insert(prefix.end(), before[i].ids.begin(), before[i].ids.end());
            }
            clip.insert(clip.end(), pcm.begin(), pcm.end());
            P = ears.encode(clip.data(), (int) clip.size(), emb);
            te = secs(t0);
            brain.reset();
            brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
            brain.step_batch(nullptr, emb.data(), P, false);
            brain.step_batch(post.data(), nullptr, (int) post.size(), prefix.empty());
            if (!prefix.empty()) brain.step_batch(prefix.data(), nullptr, (int) prefix.size(), true);
            n_read += (int) (pre.size() + P + post.size() + prefix.size());
        }
        n_prefix_all += (int) prefix.size();
        n_read_all += n_read;
        const double tr = secs(t0);
        std::vector<int32_t> ids;
        for (int i = 0; i < 512; i++) {
            int best = 0;
            for (int v = 1; v < brain.n_vocab; v++) if (brain.logits[v] > brain.logits[best]) best = v;
            if (best == words.im_end) break;
            ids.push_back(best);
            brain.step(best, nullptr, true);
        }
        if (keep_s > 0) before.push_back({ pcm, emb, P, trimmed(ids) });
        else if (ctx_s > 0) before.push_back({ pcm, {}, 0, trimmed(ids) });
        const double t = secs(t0);
        sound_all += snd; t_all += t; t_ears += te; t_read += tr - te; t_gen += t - tr; n_gen_all += (int) ids.size();
        std::string text = words.text(ids.data(), (int) ids.size());
        while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
        const std::string out = path.substr(0, path.size() - 4) + ".ko.txt";
        FILE * o = fopen(out.c_str(), "w"); fputs(text.c_str(), o); fclose(o);
        printf("%s %.2f s (+%.2f s before, %zu words given), %zu words, %.0f ms [%s]\n", path.c_str(), snd, have, prefix.size(), ids.size(), 1e3 * t, text.c_str());
    }
    if (keep_s > 0) fprintf(stderr, "imi-hear-pieces: kept memory up to %.1f s, starting over from %.1f s, %d restarts%s\n", keep_s, keep_o, restarts,
                            keep_check ? (", long-way check: worst word-score gap " + std::to_string(check_worst)).c_str() : "");
    if (turns) fprintf(stderr, "imi-hear-pieces: chat turns, %d restarts\n", restarts);
    fprintf(stderr, "imi-hear-pieces: %d pieces, %.1f s of sound in %.1f s -> %.3f CPU-s per sound second (context %.1f s, %d words given in all)\n",
            argc - 4, sound_all, t_all, t_all / sound_all, ctx_s, n_prefix_all);
    fprintf(stderr, "imi-hear-pieces: time split: ears %.1f s, reading %.1f s (%d positions), writing %.1f s (%d words, %.1f ms each)\n",
            t_ears, t_read, n_read_all, t_gen, n_gen_all, 1e3 * t_gen / std::max(1, n_gen_all));
    return 0;
}
