// imi-instant: research, the instant interpreter on a clock. The speaker's audio comes in at its real pace, cut at his
// pauses (pause_cut.py's pieces); each piece is heard as soon as it is in, each Korean sentence is translated as soon as
// its final verb ending lands, and each English sentence is spoken as soon as it is translated.
//   hearing      the memory keeps the audio so far (up to 12 s, then it starts over from the last 4 s) and goes back to
//                its end after every answer, so no audio is heard twice (imi-hear-pieces IMI_KEEP_S=12)
//   translating  sentence ends (korean.h); the memory keeps the Korean so far and goes back to its end, the English
//                already said is given back as the answer's start (imi-translate-pieces IMI_KEEP_N=4)
//   voice        each English sentence from a fresh "tts" prompt (imi-speak IMI_SENTENCES=fresh)
// With RIDE 1, every hearing or translating word rides on a voice frame's brain step (step2: one table read, two
// conversations; the same numbers as stepping alone). Reading a prompt or audio can't ride; while the voice speaks, a
// read is taken as soon as the speech already made covers the read's guessed time (12 ms a position), so speech that
// has started never stalls for it. Order: start the next English sentence, then the oldest work first (translate
// a sentence already heard before hearing the next piece).
// The clock is real: the program waits for a piece's time to come. A piece is in 0.2 s after its cut (a cut sits in
// the middle of a pause of at least 0.25 s). The English is placed on the same clock as it would be heard: each voice
// frame (80 ms) plays at its making time or right after the frame before, whichever is later.
// IMI_HEAR_AHEAD=1: while he is still talking, the piece's audio so far goes into the hearing memory (every 12 final
//   rows, the ears' last row held back), so at his pause only the rest of it and the answer are left.
// IMI_NO_VOICE=1: captions only (no voice); a sentence's lag is then the end of its Korean to its English written.
// A cuts line may carry a 4th number, when the cut was known (pause_live.py: the moment the pause is long enough);
//   the piece is in then. Without it, 0.2 s after the cut.
// The voice's lead-in: while nothing is playing (the listener is waiting for this sentence), a sentence's quiet start
//   isn't played: it begins 60 ms before its first sound. After a sentence still playing, the lead-in stays as the
//   breath between them. IMI_LEAD=0 plays every lead-in.
// The voice's prompt is read ahead (IMI_VP=0: not): the voice's fixed head is read once, and the English goes into a
//   second voice memory (VP) word by word as it is translated, each word riding a translating step (step2), so at
//   its turn only the prompt's last tokens are left to read. IMI_VP_CHECK=1 compares it with a fresh read.
//   imi-instant MODEL.gguf MMPROJ.gguf OUT_DIR threads RIDE PIECES_DIR NAME [NAME ...]
// PIECES_DIR holds NAME-KK.f32 and NAME.cuts ("KK start end" seconds per piece). Each NAME runs on its own, memories
// fresh. Writes OUT_DIR/NAME.instant.tsv (one line per sentence), NAME.ko.txt, NAME.en.txt, NAME.live.wav (the English
// as heard, on the speaker's clock) and prints the lag of every sentence: the end of its Korean to the start of its
// English.
#include "ears.h"
#include "korean.h"
#include "seoul.h"
#include "talk.h"
#include "words.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static std::string chat_head(const std::string & sys) { return "<|startoftext|><|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n"; }
static const char * CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think></think>";

struct piece { std::vector<float> pcm; double start = 0, end = 0, avail = 0; };
struct sentence {
    std::string ko, en;
    double ko_s = 0, ko_end = 0;                 // seconds of speech in it, where its Korean ended (speaker's clock)
    double tr_start = 0, tr_done = 0;            // translating, on the clock
    double play_start = -1, play_end = 0;        // its English as heard
    int frames = 0, gaps = 0;
};

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: imi-instant MODEL.gguf MMPROJ.gguf OUT_DIR threads RIDE PIECES_DIR NAME [NAME ...]\n"); return 1; }
    const std::string dir = argv[3], pdir = argv[6];
    const int threads = atoi(argv[4]);
    const bool ride = atoi(argv[5]) != 0;
    const double keep_s = 12, keep_o = 4, cut_delay = 0.2, read_s = 0.012, margin_s = 0.15;
    const int keep_n = 4, n_ctx = 4096;
    const bool no_voice = getenv("IMI_NO_VOICE") && atoi(getenv("IMI_NO_VOICE")) > 0;
    const bool hear_ahead = getenv("IMI_HEAR_AHEAD") && atoi(getenv("IMI_HEAR_AHEAD")) > 0;
    imi_team team;
    team.start(threads);
    imi_seoul brain;
    imi_words words;
    imi_ears ears;
    imi_voice voice;
    imi_codec codec;
    if (!brain.load(argv[1], threads, n_ctx, &team) || !words.load(brain.f) || !ears.load(argv[2], threads, &team) ||
        !voice.load(argv[2], threads, &team) || !codec.load(argv[2], threads, &team)) return 1;
    const std::vector<int32_t> pre = words.encode_prompt(chat_head("Perform ASR in Korean.") + "<|vision_start|>"),
                               post = words.encode_prompt(std::string("<|vision_end|>") + CHAT_TAIL),
                               tr_head = words.encode_prompt(chat_head("Please translate the following Korean into English text.")),
                               tr_tail = words.encode_prompt(CHAT_TAIL);
    auto trimmed = [&](std::vector<int32_t> ids) {   // without a trailing line break or space, so the next words start themselves
        while (!ids.empty()) {
            const std::string t = words.text(&ids.back(), 1);
            if (t.find_first_not_of(" \n") != std::string::npos) break;
            ids.pop_back();
        }
        return ids;
    };
    auto best_of = [&](const std::vector<float> & logits) {
        int best = 0;
        for (int v = 1; v < brain.n_vocab; v++) if (logits[v] > logits[best]) best = v;
        return best;
    };

    double all_sound = 0, all_busy = 0;
    std::vector<double> all_lags;
    int all_gaps = 0, all_ride = 0, all_text_alone = 0, all_voice_alone = 0;
    for (int a = 7; a < argc; a++) {
        const std::string name = argv[a];
        // the pieces and when each is in
        std::vector<piece> pieces;
        {
            std::ifstream cuts(pdir + "/" + name + ".cuts");
            std::string line;
            while (std::getline(cuts, line)) {
                std::istringstream ls(line);
                std::string kk; double s, e, ready = -1;
                if (!(ls >> kk >> s >> e)) continue;
                if (!(ls >> ready)) ready = -1;
                piece p; p.start = s; p.end = e; p.avail = ready >= 0 ? ready : e + cut_delay;
                FILE * f = fopen((pdir + "/" + name + "-" + kk + ".f32").c_str(), "rb");
                if (!f) { fprintf(stderr, "no piece %s-%s\n", name.c_str(), kk.c_str()); return 1; }
                float x; while (fread(&x, 4, 1, f) == 1) p.pcm.push_back(x);
                fclose(f);
                pieces.push_back(std::move(p));
            }
        }
        if (pieces.empty()) continue;
        const double sound_s = pieces.back().end;
        pieces.back().avail = pieces.back().end;   // the end of the stretch: no pause to wait for

        // hearing: its own memory (H), going back to the end of the audio after every answer
        imi_seoul_state H = brain.new_state();
        struct earlier { std::vector<float> emb; int P = 0; double secs = 0; std::vector<int32_t> ids; };
        std::vector<earlier> heard;   // the pieces whose audio H holds
        bool h_open = false;
        int h_snap_pos = 0;
        std::vector<std::vector<float>> h_snap_conv;
        double h_kept = 0;
        size_t next_piece = 0;        // the next piece to hear
        bool h_busy = false;
        bool h_begun = false;         // the memory is open for the coming piece
        int h_part = 0;               // rows of the coming piece already in the memory
        int n_ahead = 0;
        int h_word = -1;              // the next hearing word to step (-1: none waiting)
        std::vector<int32_t> h_ids;
        std::vector<float> h_emb; int h_P = 0;
        // translating: its own memory (X), going back to the end of the Korean after every answer
        imi_seoul_state X = brain.new_state();
        struct said { std::string ko; std::vector<int32_t> en; };
        std::vector<said> kept;
        bool x_open = false;
        int x_snap_pos = 0;
        std::vector<std::vector<float>> x_snap_conv;
        std::deque<size_t> to_translate;
        bool x_busy = false;
        size_t x_sent = 0;
        int x_word = -1;
        std::vector<int32_t> x_ids;
        // the sentence being heard, and every sentence
        std::vector<sentence> sents;
        std::string ko_now;
        double ko_now_s = 0;
        // voice: the brain's own memory
        std::deque<size_t> to_speak;
        imi_talker tk(brain, voice, codec);
        bool speaking = false;
        size_t v_sent = 0;
        std::vector<float> fb(voice.n_in), pcm, live;   // live: the English as heard, 24 kHz
        double cursor = 0;            // when the speech made so far ends, as heard
        int n_ride = 0, n_text_alone = 0, n_voice_alone = 0, n_vp_ride = 0;
        double busy = 0;
        bool v_lead = false;          // the sentence's quiet lead-in is being held back
        std::vector<float> v_held;
        // the voice's head in its own memory and in VP; a fresh sentence goes back to the end of it
        static const bool vp_on = !getenv("IMI_VP") || atoi(getenv("IMI_VP")) != 0;
        static const bool vp_check = getenv("IMI_VP_CHECK") && atoi(getenv("IMI_VP_CHECK")) > 0;
        static const bool lead_on = !getenv("IMI_LEAD") || atoi(getenv("IMI_LEAD")) != 0;
        const std::vector<int32_t> v_head = words.encode_prompt(imi_tts_head());
        imi_seoul_state VP = brain.new_state();
        brain.reset();
        brain.step_batch(v_head.data(), nullptr, (int) v_head.size(), false);
        const std::vector<std::vector<float>> v_head_conv = brain.conv_state;
        brain.swap_state(VP);
        brain.reset();
        brain.step_batch(v_head.data(), nullptr, (int) v_head.size(), false);
        brain.swap_state(VP);
        tk.set_head(v_head, v_head_conv);
        int vp_sent = -1;             // the sentence VP is reading (-1: none)
        bool vp_bad = false;          // its words stopped matching: read it from the head at its turn
        std::vector<int32_t> vp_plan; // its first English sentence so far, as tokens that can't change any more
        size_t vp_done = 0;           // of vp_plan, read into VP
        auto vp_begin = [&](size_t k) {
            VP.pos = (int) v_head.size();
            VP.conv_state = v_head_conv;
            vp_sent = (int) k; vp_bad = false; vp_plan.clear(); vp_done = 0;
        };

        const auto t0 = clk::now();
        auto now = [&] { return std::chrono::duration<double>(clk::now() - t0).count(); };
        auto in = [&](imi_seoul_state & S, auto fn) { brain.swap_state(S); fn(); brain.swap_state(S); };

        // the memory opens for pieces[next_piece] (or goes back to the end of the audio so far); past keep_s seconds of
        // audio it starts over from the last few seconds of pieces. With hearing ahead the piece's length isn't known
        // yet, so it starts over when less than 3 s of room is left.
        auto hear_begin = [&](double secs) {
            if (h_open && (hear_ahead ? h_kept > keep_s - 3 : h_kept + secs > keep_s)) {
                size_t first = heard.size();
                double o = 0;
                while (first > 0 && o + heard[first - 1].secs <= keep_o) { first--; o += heard[first].secs; }
                heard.erase(heard.begin(), heard.begin() + first);
                h_open = false;
            }
            in(H, [&] {
                if (!h_open) {
                    brain.reset();
                    brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
                    h_kept = 0;
                    for (auto & e : heard) { brain.step_batch(nullptr, e.emb.data(), e.P, false); h_kept += e.secs; }
                    h_open = true;
                    h_snap_pos = brain.pos;
                    h_snap_conv = brain.conv_state;
                }
            });
            h_part = 0;
            h_begun = true;
        };
        // rows [from, to) of the coming piece go into the memory, after the audio so far
        auto hear_rows = [&](const std::vector<float> & emb, int from, int to) {
            if (to <= from) return;
            in(H, [&] {
                brain.pos = h_snap_pos;
                brain.conv_state = h_snap_conv;
                brain.step_batch(nullptr, emb.data() + (size_t) from * brain.n_embd, to - from, false);
                h_snap_pos = brain.pos;
                h_snap_conv = brain.conv_state;
            });
            h_kept += (to - from) * 0.08;
            h_part = to;
        };
        // hearing ahead: while he is still talking, the piece's audio so far goes in; the ears' last row is held
        // back (it changes as more audio comes; every row before it is final, imi-ears-causal)
        auto ahead_rows = [&](double t) {
            if (!hear_ahead || h_busy || next_piece >= pieces.size()) return 0;
            const piece & p = pieces[next_piece];
            if (t >= p.avail) return 0;
            const double have = std::min(t, p.end) - p.start;
            const int final_rows = (int) (have * 16000 / 1280) - 1;
            return final_rows - h_part >= 12 ? final_rows : 0;
        };
        auto hear_ahead_step = [&](double t) {
            const piece & p = pieces[next_piece];
            const size_t n = std::min(p.pcm.size(), (size_t) ((std::min(t, p.end) - p.start) * 16000));
            if (!h_begun) hear_begin(0);
            std::vector<float> emb;
            const int P = ears.encode(p.pcm.data(), (int) n, emb);
            hear_rows(emb, h_part, P - 1);
            n_ahead++;
        };
        // the piece is in: the rest of its audio goes in, then the answer starts with the Korean already written
        auto hear_start = [&] {
            const piece & p = pieces[next_piece];
            const double secs = p.pcm.size() / 16000.0;
            if (!h_begun) hear_begin(secs);
            h_P = ears.encode(p.pcm.data(), (int) p.pcm.size(), h_emb);
            hear_rows(h_emb, h_part, h_P);
            std::vector<int32_t> prefix;
            for (auto & e : heard) prefix.insert(prefix.end(), e.ids.begin(), e.ids.end());
            in(H, [&] {
                brain.pos = h_snap_pos;
                brain.conv_state = h_snap_conv;
                brain.step_batch(post.data(), nullptr, (int) post.size(), prefix.empty());
                if (!prefix.empty()) brain.step_batch(prefix.data(), nullptr, (int) prefix.size(), true);
            });
            h_ids.clear();
            h_busy = true;
        };
        auto translate_start = [&] {
            x_sent = to_translate.front();
            to_translate.pop_front();
            sentence & s = sents[x_sent];
            s.tr_start = now();
            if (x_open && (int) kept.size() >= keep_n) {   // start over from the last 2 sentences
                kept.erase(kept.begin(), kept.end() - std::min<size_t>(2, kept.size()));
                x_open = false;
            }
            auto ko_ids = [&](size_t i, const std::string & ko) { return words.encode_prompt((i == 0 ? "" : " ") + ko); };
            in(X, [&] {
                if (!x_open) {
                    brain.reset();
                    std::vector<int32_t> tp = tr_head;
                    for (size_t i = 0; i < kept.size(); i++) { const auto k = ko_ids(i, kept[i].ko); tp.insert(tp.end(), k.begin(), k.end()); }
                    brain.step_batch(tp.data(), nullptr, (int) tp.size(), false);
                    x_open = true;
                } else {
                    brain.pos = x_snap_pos;
                    brain.conv_state = x_snap_conv;
                }
                const std::vector<int32_t> nk = ko_ids(kept.size(), s.ko);
                brain.step_batch(nk.data(), nullptr, (int) nk.size(), false);
                x_snap_pos = brain.pos;
                x_snap_conv = brain.conv_state;
                std::vector<int32_t> given = tr_tail;
                for (auto & c : kept) given.insert(given.end(), c.en.begin(), c.en.end());
                brain.step_batch(given.data(), nullptr, (int) given.size(), true);
            });
            x_ids.clear();
            x_busy = true;
            if (vp_on && !no_voice && vp_sent < 0) vp_begin(x_sent);
        };
        // after a hearing step (or its reads): the next word, or the piece is done
        auto hear_pick = [&] {
            const int w = best_of(H.logits);
            if (w != words.im_end && h_ids.size() < 512) { h_ids.push_back(w); h_word = w; return; }
            h_word = -1;
            h_busy = false;
            const piece & p = pieces[next_piece];
            heard.push_back({ h_emb, h_P, p.pcm.size() / 16000.0, trimmed(h_ids) });
            h_begun = false;
            h_part = 0;
            const std::string ko = imi_ko_piece(words.text(h_ids.data(), (int) h_ids.size()));
            if (getenv("IMI_PIECE_LOG")) fprintf(stderr, "  piece %zu %.2f-%.2f heard at %.2f: [%s]\n", next_piece, p.start, p.end, now(), ko.c_str());
            ko_now_s += p.end - p.start;
            if (!ko.empty()) ko_now += (ko_now.empty() ? "" : " ") + ko;
            // a sentence end inside: the words before it are a sentence of their own, translated now
            for (size_t e; (e = imi_ko_inner_end(ko_now)) != std::string::npos; ) {
                sentence s; s.ko = ko_now.substr(0, e);
                s.ko_s = ko_now_s * (double) e / ko_now.size(); s.ko_end = p.end;
                sents.push_back(s);
                to_translate.push_back(sents.size() - 1);
                ko_now = ko_now.substr(ko_now.find_first_not_of(' ', e));
                ko_now_s -= s.ko_s;
            }
            const bool last = next_piece + 1 == pieces.size();
            if (!ko_now.empty() && (imi_ko_translate_now(ko_now, ko_now_s) || last)) {
                sentence s; s.ko = ko_now; s.ko_s = ko_now_s; s.ko_end = p.end;
                sents.push_back(s);
                to_translate.push_back(sents.size() - 1);
                ko_now.clear();
                ko_now_s = 0;
            }
            next_piece++;
        };
        // VP's plan: the English so far, up to its first sentence's end if that is in, else up to the last whole word
        auto vp_update = [&] {
            if (vp_sent != (int) x_sent || vp_bad) return;
            std::string t = words.text(x_ids.data(), (int) x_ids.size());
            const size_t a = t.find_first_not_of(" \t\n");
            if (a == std::string::npos) return;
            t = t.substr(a);
            const std::vector<std::string> ss = imi_sentences(t);
            std::string stable;
            if (ss.size() >= 2) stable = ss[0];
            else {
                const size_t sp = t.find_last_of(" \t\n");
                if (sp == std::string::npos) return;
                stable = t.substr(0, sp);
                const size_t b = stable.find_last_not_of(" \t\n");
                if (b == std::string::npos) return;
                stable = stable.substr(0, b + 1);
            }
            const std::vector<int32_t> plan = words.encode_prompt(stable);
            size_t c = 0;
            while (c < plan.size() && c < vp_plan.size() && plan[c] == vp_plan[c]) c++;
            if (c < vp_done) { vp_bad = true; return; }
            vp_plan = plan;
        };
        auto translate_pick = [&] {
            const int w = best_of(X.logits);
            if (w != words.im_end && x_ids.size() < 300) { x_ids.push_back(w); x_word = w; vp_update(); return; }
            x_word = -1;
            x_busy = false;
            sentence & s = sents[x_sent];
            const std::vector<int32_t> en = trimmed(x_ids);
            kept.push_back({ s.ko, en });
            s.en = words.text(en.data(), (int) en.size());
            while (!s.en.empty() && s.en.front() == ' ') s.en.erase(s.en.begin());
            s.tr_done = now();
            if (!no_voice && s.en.find_first_not_of(" \n") != std::string::npos) to_speak.push_back(x_sent);
        };
        auto speak_start = [&] {
            v_sent = to_speak.front();
            to_speak.pop_front();
            bool read = true;
            if (vp_on) {
                const std::string & en = sents[v_sent].en;
                std::vector<std::string> ss = imi_sentences(en);
                if (ss.empty()) ss.push_back(en);
                const std::vector<int32_t> full = words.encode_prompt(imi_tts_prompt(ss[0]));
                const size_t h = v_head.size();
                if (full.size() > h && std::equal(v_head.begin(), v_head.end(), full.begin())) {
                    const bool mine = vp_sent == (int) v_sent && !vp_bad && full.size() > h + vp_done &&
                                      std::equal(vp_plan.begin(), vp_plan.begin() + vp_done, full.begin() + h);
                    if (!mine) { VP.pos = (int) h; VP.conv_state = v_head_conv; vp_done = 0; }
                    in(VP, [&] { brain.step_batch(full.data() + h + vp_done, nullptr, (int) (full.size() - h - vp_done), false); });
                    if (vp_check) {   // the same prompt read fresh, in one go
                        imi_seoul_state F = brain.new_state();
                        std::vector<float> got = VP.hidden;
                        in(F, [&] { brain.reset(); brain.step_batch(full.data(), nullptr, (int) full.size(), false); });
                        double d = 0;
                        for (size_t i = 0; i < got.size(); i++) d = std::max(d, (double) std::fabs(got[i] - F.hidden[i]));
                        fprintf(stderr, "imi-vp-check: sentence %zu, %zu of %zu English tokens read ahead, largest gap %.3e\n", v_sent, mine ? vp_done : 0, full.size() - h, d);
                    }
                    brain.swap_state(VP);   // VP's memory becomes the voice's; the old voice memory is free for the next one
                    read = false;
                }
                vp_sent = -1; vp_bad = false; vp_plan.clear(); vp_done = 0;
                if (x_busy) vp_begin(x_sent);
            }
            tk.begin_sentences(words, sents[v_sent].en, 11 + (unsigned) v_sent, 2000, false, read);
            speaking = true;
            v_lead = true;
            v_held.clear();
        };
        // one voice frame; its brain step rides with a text word if one is waiting
        auto voice_frame = [&] {
            const size_t before = pcm.size();
            speaking = tk.next(pcm, fb.data());
            sentence & s = sents[v_sent];
            const double made = now();
            std::vector<float> got(pcm.begin() + before, pcm.end());
            if (!got.empty() && lead_on && v_lead && cursor <= made) {
                // nothing is playing: hold the quiet start back, begin 60 ms before the first 20 ms over -50 dB
                v_held.insert(v_held.end(), got.begin(), got.end());
                size_t first = v_held.size();
                for (size_t i = 0; i + 480 <= v_held.size(); i += 480) {
                    double e = 0;
                    for (size_t j = i; j < i + 480; j++) e += (double) v_held[j] * v_held[j];
                    if (10 * std::log10(e / 480 + 1e-12) > -50) { first = i; break; }
                }
                const size_t keep = 1440;
                if (first == v_held.size()) {
                    if (v_held.size() > keep) v_held.erase(v_held.begin(), v_held.end() - keep);
                    got.clear();
                } else {
                    got.assign(v_held.begin() + (first > keep ? first - keep : 0), v_held.end());
                    v_held.clear();
                    v_lead = false;
                }
            } else if (!got.empty()) v_lead = false;
            if (!got.empty()) {   // place the new samples on the clock as heard
                const double play = std::max(made, cursor);
                if (s.play_start < 0) s.play_start = play;
                else if (play > cursor + 1e-3) s.gaps++;
                const size_t at = (size_t) (play * 24000);
                const size_t n = got.size();
                if (live.size() < at + n) live.resize(at + n, 0.0f);
                std::copy(got.begin(), got.end(), live.begin() + at);
                cursor = play + n / 24000.0;
                s.play_end = cursor;
            }
            if (!speaking) { s.frames = tk.st.frames; return; }
            if (ride && x_word >= 0) { brain.step2(-1, fb.data(), false, X, x_word, nullptr, true); n_ride++; translate_pick(); return; }
            if (ride && h_word >= 0) { brain.step2(-1, fb.data(), false, H, h_word, nullptr, true); n_ride++; hear_pick(); return; }
            brain.step(-1, fb.data(), false);
            n_voice_alone++;
        };

        // a read's time, guessed from its positions (about 12 ms each on the floor box, the ears about 35 ms per second
        // of audio): a read is taken while speaking only when the speech ready covers it, so speech never stalls
        auto hear_est = [&] {
            const piece & p = pieces[next_piece];
            int n = (int) (p.pcm.size() / 1280) - h_part + (int) post.size();
            for (auto & e : heard) n += (int) e.ids.size() + (h_open ? 0 : e.P);
            if (!h_open) n += (int) pre.size();
            return n * read_s + 0.035 * p.pcm.size() / 16000.0;
        };
        auto translate_est = [&] {
            int n = (int) tr_tail.size() + (int) sents[to_translate.front()].ko.size() / 3;
            for (auto & c : kept) n += (int) c.en.size() + (x_open ? 0 : (int) c.ko.size() / 3);
            return n * read_s;
        };
        static const bool sched_log = getenv("IMI_SCHED_LOG") != nullptr;
        for (;;) {
            const double t = now();
            const bool piece_in = !h_busy && next_piece < pieces.size() && pieces[next_piece].avail <= t;
            const bool tr_in = !x_busy && !to_translate.empty();
            const double ahead = speaking ? cursor - t : 0;
            const auto w0 = clk::now();
            if (!speaking && !to_speak.empty()) {
                speak_start();
            } else if (tr_in && (!speaking || ahead >= translate_est() + margin_s)) {
                translate_start(); translate_pick();
            } else if (piece_in && (!speaking || ahead >= hear_est() + margin_s)) {
                if (sched_log) fprintf(stderr, "  sched %.3f hear piece %zu (rows ahead %d)\n", t, next_piece, h_part);
                hear_start(); hear_pick();
                if (sched_log) fprintf(stderr, "  sched %.3f  read done\n", now());
            } else if (ahead_rows(t) > 0 && (!speaking || ahead >= 12 * read_s + 0.035 * (t - pieces[next_piece].start) + margin_s)) {
                if (sched_log) fprintf(stderr, "  sched %.3f ahead piece %zu rows %d..%d\n", t, next_piece, h_part, ahead_rows(t));
                hear_ahead_step(t);
            } else if (speaking) {
                voice_frame();
            } else if (x_word >= 0) {
                const int w = x_word;
                if (vp_sent == (int) x_sent && !vp_bad && vp_done < vp_plan.size()) {   // an English word rides into VP
                    const int32_t vt = vp_plan[vp_done++];
                    in(X, [&] { brain.step2(w, nullptr, true, VP, vt, nullptr, false); });
                    n_vp_ride++;
                } else in(X, [&] { brain.step(w, nullptr, true); });
                n_text_alone++; translate_pick();
            } else if (h_word >= 0) {
                const int w = h_word; in(H, [&] { brain.step(w, nullptr, true); }); n_text_alone++; hear_pick();
            } else if (next_piece >= pieces.size() && !h_busy && !x_busy && to_translate.empty() && to_speak.empty()) {
                break;
            } else {
                const double wait = next_piece < pieces.size() ? pieces[next_piece].avail - t : 0.005;
                std::this_thread::sleep_for(std::chrono::duration<double>(std::max(0.001, std::min(wait, 0.05))));
                continue;
            }
            busy += std::chrono::duration<double>(clk::now() - w0).count();
        }

        // what happened, sentence by sentence
        std::ofstream ts(dir + "/" + name + ".instant.tsv");
        ts << "ko_end\tko_s\ttr_start\ttr_done\tplay_start\tplay_end\tlag\tframes\tgaps\tko\ten\n";
        std::string ko_all, en_all;
        std::vector<double> lags;
        int gaps = 0;
        for (auto & s : sents) {
            const double lag = no_voice ? s.tr_done - s.ko_end : s.play_start >= 0 ? s.play_start - s.ko_end : -1;
            if (no_voice || s.play_start >= 0) lags.push_back(lag);
            gaps += s.gaps;
            char b[160];
            snprintf(b, sizeof b, "%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t", s.ko_end, s.ko_s, s.tr_start, s.tr_done, s.play_start, s.play_end, lag, s.frames, s.gaps);
            ts << b << s.ko << "\t" << s.en << "\n";
            ko_all += (ko_all.empty() ? "" : " ") + s.ko;
            en_all += (en_all.empty() ? "" : " ") + s.en;
            fprintf(stderr, "  %6.2f s  lag %5.2f s  [%s] -> [%s]\n", s.ko_end, lag, s.ko.c_str(), s.en.c_str());
        }
        { std::ofstream f(dir + "/" + name + ".ko.txt"); f << ko_all; }
        { std::ofstream f(dir + "/" + name + ".en.txt"); f << en_all; }
        imi_write_wav(dir + "/" + name + ".live.wav", live, 24000);
        std::vector<double> sl = lags;
        std::sort(sl.begin(), sl.end());
        const double med = sl.empty() ? 0 : sl[sl.size() / 2], worst = sl.empty() ? 0 : sl.back();
        fprintf(stderr, "imi-instant: %s %.1f s of sound, %zu sentences, lag median %.2f s worst %.2f s, %d gaps in speech, busy %.1f s -> %.3f per sound second, speech ends %.1f s after his (ride %d: %d shared, %d text alone, %d voice alone; %d reads ahead; %d English words read ahead)\n",
                name.c_str(), sound_s, sents.size(), med, worst, gaps, busy, busy / sound_s, cursor - sound_s, ride ? 1 : 0, n_ride, n_text_alone, n_voice_alone, n_ahead, n_vp_ride);
        all_sound += sound_s; all_busy += busy; all_gaps += gaps;
        all_lags.insert(all_lags.end(), lags.begin(), lags.end());
        all_ride += n_ride; all_text_alone += n_text_alone; all_voice_alone += n_voice_alone;
    }
    std::sort(all_lags.begin(), all_lags.end());
    auto q = [&](double f) { return all_lags.empty() ? 0.0 : all_lags[std::min(all_lags.size() - 1, (size_t) (f * all_lags.size()))]; };
    fprintf(stderr, "imi-instant: all %.1f s of sound, %zu sentences, lag median %.2f s, 90%% %.2f s, worst %.2f s, %d gaps, busy %.3f per sound second (ride %d: %d shared, %d text alone, %d voice alone)\n",
            all_sound, all_lags.size(), q(0.5), q(0.9), all_lags.empty() ? 0.0 : all_lags.back(), all_gaps, all_busy / all_sound, ride ? 1 : 0,
            all_ride, all_text_alone, all_voice_alone);
    return 0;
}
