// imi-translate-pieces: research, translating as the Korean comes in. Heard pieces (cut at the speaker's pauses) wait
// until the Korean so far ends a sentence, then that sentence is translated.
//   imi-translate-pieces MODEL.gguf threads PIECES.tsv OUT.tsv [CTX_CLAUSES=3]
// PIECES.tsv: "name<TAB>korean[<TAB>seconds]" per piece in order; a new name starts fresh (no earlier sentences).
// OUT.tsv: "name<TAB>korean sentence<TAB>english<TAB>ms<TAB>seconds of speech" per sentence.
//
// When to translate (IMI_CUT):
//   sentence (default)  the Korean ends in a sentence-final verb ending (-다 -요 -죠 -라 -오, -까 as a question, or ? !).
//                       Korean puts the verb last, so a sentence is only known once its last ending lands. A sentence
//                       that runs past 8 s is cut at a joining ending (-고 -서 -며 -면 -데 -지만 ...), past 12 s anywhere.
//   joins               the first rule tried: final or joining endings, or 3 pieces / 180 bytes waited.
// The hearing writes a full stop at the end of every piece, also mid-sentence, where the speaker only paused. That stop
// is taken out unless the piece really ends a sentence.
//
// How earlier sentences are given (IMI_TURNS):
//   unset  one prompt holds the earlier CTX_CLAUSES sentences' Korean and the new one together; the English already
//          said goes in as the start of the answer, so the model writes only the new English. All of it is read again
//          for every sentence.
//   1      a chat: each sentence is a new user turn in the same memory, so nothing is read twice. When the memory is
//          nearly full it starts over with the last 3 turns read back in. (Read by hand: it sometimes answers the
//          speaker's question instead of translating it, "The commonality is God's sovereignty.")
// IMI_KEEP_N=N: the first layout (Korean together, English given back), but the Korean is read once. The memory keeps
//   the Korean so far and after each answer goes back to the end of it (attention caches are written by position, so
//   going back is setting the position and the conv memory back); the next sentence's Korean goes in there, then the
//   answer starts again with the English already said. After N sentences it starts over from the last 2.
//   IMI_KEEP_CHECK=1 also reads every sentence the long way in a second memory and prints the worst word-score gap.
#include "korean.h"
#include "seoul.h"
#include "words.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::string chat_head(const std::string & sys) { return "<|startoftext|><|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n"; }
static const char * CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think></think>";
static const char * TURN = "<|im_end|>\n<|im_start|>user\n";   // after an answer's last word: the next user turn

static bool old_rule(const std::string & ko) {   // the first rule: final or joining endings
    const imi_ko_tail t = imi_ko_tail_of(ko);
    if (t.mark) return true;
    if (t.syl.empty()) return false;
    for (const char * e : { "다", "요", "까", "죠", "오", "고", "서", "며", "면", "데", "만", "니", "지" }) if (t.syl.back() == imi_ko_syl(e)) return true;
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: imi-translate-pieces MODEL.gguf threads PIECES.tsv OUT.tsv [CTX_CLAUSES]\n"); return 1; }
    const int threads = atoi(argv[2]);
    const int ctx_n = argc > 5 ? atoi(argv[5]) : 3;
    const bool joins = getenv("IMI_CUT") && std::string(getenv("IMI_CUT")) == "joins";
    const bool turns = getenv("IMI_TURNS") && atoi(getenv("IMI_TURNS")) > 0;
    const int keep_n = getenv("IMI_KEEP_N") ? atoi(getenv("IMI_KEEP_N")) : 0;
    const bool keep_check = getenv("IMI_KEEP_CHECK") && atoi(getenv("IMI_KEEP_CHECK")) > 0;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    imi_team team;
    team.start(threads);
    imi_seoul brain;
    const int n_ctx = 4096;
    if (!brain.load(argv[1], threads, n_ctx, &team)) return 1;
    imi_words words;
    if (!words.load(brain.f)) return 1;
    const std::string head = chat_head("Please translate the following Korean into English text.");

    std::ifstream in(argv[3]);
    FILE * out = fopen(argv[4], "w");
    std::string line, name;
    struct clause { std::string ko; std::vector<int32_t> en; };
    std::vector<clause> done;       // this run's translated sentences
    std::string pending;            // heard Korean not translated yet
    double pending_s = 0;
    int pending_pieces = 0, n_clauses = 0, n_words = 0, n_read = 0, n_empty = 0, n_restarts = 0;
    double t_all = 0, t_read = 0;   // research: reading (prompt + English given) vs writing
    bool chat_open = false;         // IMI_TURNS: the memory holds this run's chat so far
    std::vector<clause> kept;       // IMI_KEEP_N: the sentences whose Korean the memory holds
    bool k_open = false;
    int snap_pos = 0;               // IMI_KEEP_N: the end of the Korean so far
    std::vector<std::vector<float>> snap_conv;
    double check_worst = 0;
    imi_seoul_state second;         // IMI_KEEP_CHECK: the long way's memory
    if (keep_check) second = brain.new_state();
    const std::vector<int32_t> head_ids = words.encode_prompt(head), tail_ids = words.encode_prompt(CHAT_TAIL);
    auto ko_ids = [&](size_t i, const std::string & ko) { return words.encode_prompt((i == 0 ? "" : " ") + ko); };
    auto trim_end = [&](std::vector<int32_t> & en) {   // without a trailing line break, so the next English starts itself
        while (!en.empty()) {
            const std::string t = words.text(&en.back(), 1);
            if (t.find_first_not_of(" \n") != std::string::npos) break;
            en.pop_back();
        }
    };

    auto write_answer = [&](std::vector<int32_t> & en) {
        for (int i = 0; i < 300; i++) {
            int best = 0;
            for (int v = 1; v < brain.n_vocab; v++) if (brain.logits[v] > brain.logits[best]) best = v;
            if (best == words.im_end) break;
            en.push_back(best);
            brain.step(best, nullptr, true);
        }
    };
    auto translate = [&] {
        const auto t0 = clk::now();
        std::vector<int32_t> en;
        if (turns) {
            // the chat so far stays in memory; the new sentence is one more user turn
            std::vector<int32_t> tp;
            if (chat_open && brain.pos + 600 > n_ctx) { chat_open = false; n_restarts++; }
            if (!chat_open) {
                std::string text = head;
                const size_t first = done.size() > 3 ? done.size() - 3 : 0;
                for (size_t i = first; i < done.size(); i++)
                    text += done[i].ko + CHAT_TAIL + words.text(done[i].en.data(), (int) done[i].en.size()) + TURN;
                text += pending + CHAT_TAIL;
                tp = words.encode_prompt(text);
                brain.reset();
                chat_open = true;
            } else {
                tp = words.encode_prompt(std::string(TURN) + pending + CHAT_TAIL);
            }
            brain.step_batch(tp.data(), nullptr, (int) tp.size(), true);
            t_read += secs(t0); n_read += (int) tp.size();
            write_answer(en);   // its closing <|im_end|> goes in with the next turn (TURN starts with it)
        } else if (keep_n > 0) {
            if (k_open && (int) kept.size() >= keep_n) {   // start over from the last 2 sentences
                kept.erase(kept.begin(), kept.end() - std::min<size_t>(2, kept.size()));
                k_open = false;
                n_restarts++;
            }
            if (!k_open) {
                brain.reset();
                std::vector<int32_t> tp = head_ids;
                for (size_t i = 0; i < kept.size(); i++) { const auto k = ko_ids(i, kept[i].ko); tp.insert(tp.end(), k.begin(), k.end()); }
                brain.step_batch(tp.data(), nullptr, (int) tp.size(), false);
                n_read += (int) tp.size();
                k_open = true;
            } else {
                brain.pos = snap_pos;
                brain.conv_state = snap_conv;
            }
            const std::vector<int32_t> nk = ko_ids(kept.size(), pending);
            brain.step_batch(nk.data(), nullptr, (int) nk.size(), false);
            snap_pos = brain.pos;
            snap_conv = brain.conv_state;
            std::vector<int32_t> given = tail_ids;
            for (auto & c : kept) given.insert(given.end(), c.en.begin(), c.en.end());
            brain.step_batch(given.data(), nullptr, (int) given.size(), true);
            n_read += (int) (nk.size() + given.size());
            if (keep_check) {   // the long way: everything read from the start in a second memory
                brain.swap_state(second);
                brain.reset();
                std::vector<int32_t> all = head_ids;
                for (size_t i = 0; i < kept.size(); i++) { const auto k = ko_ids(i, kept[i].ko); all.insert(all.end(), k.begin(), k.end()); }
                all.insert(all.end(), nk.begin(), nk.end());
                all.insert(all.end(), given.begin(), given.end());
                brain.step_batch(all.data(), nullptr, (int) all.size(), true);
                const std::vector<float> l2 = brain.logits;
                brain.swap_state(second);
                for (int v = 0; v < brain.n_vocab; v++) check_worst = std::max(check_worst, (double) std::fabs(brain.logits[v] - l2[v]));
            }
            t_read += secs(t0);
            write_answer(en);
            trim_end(en);
            kept.push_back({ pending, en });
        } else {
            const size_t first = done.size() > (size_t) ctx_n ? done.size() - ctx_n : 0;
            std::string ko_all;
            std::vector<int32_t> given;
            for (size_t i = first; i < done.size(); i++) {
                ko_all += (ko_all.empty() ? "" : " ") + done[i].ko;
                given.insert(given.end(), done[i].en.begin(), done[i].en.end());
            }
            ko_all += (ko_all.empty() ? "" : " ") + pending;
            const std::vector<int32_t> tp = words.encode_prompt(head + ko_all + CHAT_TAIL);
            brain.reset();
            brain.step_batch(tp.data(), nullptr, (int) tp.size(), given.empty());
            if (!given.empty()) brain.step_batch(given.data(), nullptr, (int) given.size(), true);
            t_read += secs(t0); n_read += (int) (tp.size() + given.size());
            write_answer(en);
            while (!en.empty()) {   // kept without a trailing line break, so the next sentence's English starts itself
                const std::string t = words.text(&en.back(), 1);
                if (t.find_first_not_of(" \n") != std::string::npos) break;
                en.pop_back();
            }
        }
        const double t = secs(t0);
        t_all += t; n_clauses++; n_words += (int) en.size();
        const std::string en_text = words.text(en.data(), (int) en.size());
        if (en_text.find_first_not_of(" \n") == std::string::npos) n_empty++;
        fprintf(out, "%s\t%s\t%s\t%.0f\t%.1f\n", name.c_str(), pending.c_str(), en_text.c_str(), 1e3 * t, pending_s);
        fflush(out);
        done.push_back({ pending, en });
        pending.clear();
        pending_pieces = 0;
        pending_s = 0;
    };
    while (std::getline(in, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string nm = line.substr(0, tab);
        std::string ko = line.substr(tab + 1);
        double s = 0;
        const size_t tab2 = ko.find('\t');
        if (tab2 != std::string::npos) { s = atof(ko.c_str() + tab2 + 1); ko = ko.substr(0, tab2); }
        if (nm != name) {
            if (!pending.empty()) translate();
            done.clear();
            chat_open = false;
            kept.clear();
            k_open = false;
            name = nm;
        }
        if (!joins) ko = imi_ko_piece(ko);   // a pause's mark, not a sentence end, goes
        else while (!ko.empty() && (ko.back() == ' ' || ko.back() == '\n')) ko.pop_back();
        if (ko.find_first_not_of(" ") == std::string::npos) { pending_s += s; continue; }
        pending += (pending.empty() ? "" : " ") + ko;
        pending_pieces++;
        pending_s += s;
        if (joins) {
            if (old_rule(pending) || pending_pieces >= 3 || pending.size() >= 180) translate();   // 60 Hangul syllables
        } else {
            if (imi_ko_translate_now(pending, pending_s)) translate();
        }
    }
    if (!pending.empty()) translate();
    fclose(out);
    const std::string how = turns ? std::string("chat turns")
                          : keep_n > 0 ? "kept memory, " + std::to_string(keep_n) + " sentences" + (keep_check ? ", long-way check worst gap " + std::to_string(check_worst) : std::string())
                                       : "context " + std::to_string(ctx_n) + " sentences";
    fprintf(stderr, "imi-translate-pieces: %d sentences, %d English words, %d empty, %.1f s (%.0f ms per sentence), %s, cut at %s, %d restarts\n",
            n_clauses, n_words, n_empty, t_all, 1e3 * t_all / std::max(1, n_clauses), how.c_str(), joins ? "joins" : "sentence ends", n_restarts);
    fprintf(stderr, "imi-translate-pieces: time split: reading %.1f s (%d positions), writing %.1f s (%d words, %.1f ms each)\n",
            t_read, n_read, t_all - t_read, n_words, 1e3 * (t_all - t_read) / std::max(1, n_words));
    return 0;
}
