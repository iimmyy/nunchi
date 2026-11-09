#include "live.h"

#include "core/korean.h"
#include "pauses.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <sstream>
#include <thread>

using clk = std::chrono::steady_clock;

static std::string chat_head(const std::string & sys) { return "<|startoftext|><|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n"; }
static const char * CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think></think>";

bool nunchi_live::load(const std::string & model, const std::string & mmproj, int threads) {
    team.start(threads);
    return brain.load(model, threads, 4096, &team) && words.load(brain.f) && ears.load(mmproj, threads, &team) &&
           voice.load(mmproj, threads, &team) && codec.load(mmproj, threads, &team);
}

void nunchi_live::push(const float * pcm, size_t n) {
    float peak = -120;
    for (size_t i = 0; i + 320 <= n; i += 320) {
        double e = 0;
        for (size_t j = i; j < i + 320; j++) e += (double) pcm[j] * pcm[j];
        peak = std::max(peak, (float) (10 * std::log10(e / 320 + 1e-12)));
    }
    level_db = peak;
    std::lock_guard<std::mutex> l(mu);
    inbox.insert(inbox.end(), pcm, pcm + n);
}

void nunchi_live::set_terms(const std::string & lines) {
    std::vector<std::pair<std::string, std::string>> t;
    std::istringstream in(lines);
    std::string line;
    auto trim = [](std::string s) {
        const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    while (std::getline(in, line)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string ko = trim(line.substr(0, eq)), en = trim(line.substr(eq + 1));
        if (!ko.empty() && !en.empty()) t.push_back({ ko, en });
    }
    std::sort(t.begin(), t.end(), [](auto & a, auto & b) { return a.first.size() > b.first.size(); });   // longest first
    std::lock_guard<std::mutex> l(mu);
    terms = t;
    terms_raw = lines;
}

std::string nunchi_live::terms_text() { std::lock_guard<std::mutex> l(mu); return terms_raw; }

void nunchi_live::run(const nunchi_events & ev, const std::atomic<bool> & stop) {
    auto log = [&](const char * fmt, ...) {
        if (!ev.log) return;
        char b[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(b, sizeof b, fmt, ap);
        va_end(ap);
        ev.log(b);
    };
    const double keep_s = 12, keep_o = 4, read_s = 0.012, margin_s = 0.15;
    const int keep_n = 4;
    const std::vector<int32_t> pre = words.encode_prompt(chat_head("Perform ASR in Korean.") + "<|vision_start|>"),
                               post = words.encode_prompt(std::string("<|vision_end|>") + CHAT_TAIL),
                               tr_head = words.encode_prompt(chat_head("Please translate the following Korean into English text.")),
                               tr_tail = words.encode_prompt(CHAT_TAIL);
    auto trimmed = [&](std::vector<int32_t> ids) {   // without a trailing line break or space
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
    auto with_terms = [&](const std::string & ko) {   // hints in place: 사사기 -> 사사기(Judges)
        std::vector<std::pair<std::string, std::string>> t;
        { std::lock_guard<std::mutex> l(mu); t = terms; }
        std::string out;
        for (size_t i = 0; i < ko.size();) {
            bool hit = false;
            for (auto & [k, e] : t)
                if (ko.compare(i, k.size(), k) == 0) { out += k + "(" + e + ")"; i += k.size(); hit = true; break; }
            if (!hit) out += ko[i++];
        }
        return out;
    };

    const auto t0 = clk::now();
    auto now = [&] { return std::chrono::duration<double>(clk::now() - t0).count(); };
    auto in = [&](imi_seoul_state & S, auto fn) { brain.swap_state(S); fn(); brain.swap_state(S); };

    // the feed
    nunchi_pauses finder;
    std::vector<float> audio;     // the sound from sample a0 on (the open piece)
    int64_t a0 = 0, got = 0;
    bool was_listening = true;
    struct piece { std::vector<float> pcm; double start = 0, end = 0; };   // seconds of the feed
    std::deque<piece> queue;      // cut, waiting to be heard (its front is the coming piece)
    double last_piece_end = 0;

    // hearing: its own memory (H), going back to the end of the audio after every answer
    imi_seoul_state H = brain.new_state();
    struct earlier { std::vector<float> emb; int P = 0; double secs = 0; std::vector<int32_t> ids; };
    std::vector<earlier> heard;
    bool h_open = false, h_busy = false, h_begun = false;
    int h_snap_pos = 0, h_base_pos = 0, h_part = 0, h_word = -1, h_P = 0;
    std::vector<std::vector<float>> h_snap_conv, h_base_conv;
    double h_kept = 0;
    std::vector<int32_t> h_ids;
    std::vector<float> h_emb;
    // translating: its own memory (X), going back to the end of the Korean after every answer
    imi_seoul_state X = brain.new_state();
    struct said { std::string ko; std::vector<int32_t> en; };
    std::vector<said> kept;
    bool x_open = false, x_busy = false;
    int x_snap_pos = 0, x_word = -1;
    std::vector<std::vector<float>> x_snap_conv;
    std::deque<size_t> to_translate;
    size_t x_sent = 0;
    std::vector<int32_t> x_ids;
    struct sentence { std::string ko, ko_in, en; double ko_s = 0, ko_end = 0, play_start = -1; };
    std::vector<sentence> sents;
    std::string ko_now;
    double ko_now_s = 0;
    // voice: the brain's own memory; VP reads the next sentence's prompt while it is translated
    std::deque<size_t> to_speak;
    imi_talker tk(brain, voice, codec);
    bool speaking = false, v_lead = false;
    size_t v_sent = 0;
    std::vector<float> fb(voice.n_in), pcm, v_held;
    double cursor = 0;            // when the speech made so far ends at the listeners (wall seconds)
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
    int vp_sent = -1;
    bool vp_bad = false;
    std::vector<int32_t> vp_plan;
    size_t vp_done = 0;
    auto vp_begin = [&](size_t k) {
        VP.pos = (int) v_head.size();
        VP.conv_state = v_head_conv;
        vp_sent = (int) k; vp_bad = false; vp_plan.clear(); vp_done = 0;
    };

    // a sentence is ready to translate
    auto add_sentence = [&](const std::string & ko, double secs, double end) {
        sentence s; s.ko = ko; s.ko_in = with_terms(ko); s.ko_s = secs; s.ko_end = end;
        sents.push_back(s);
        to_translate.push_back(sents.size() - 1);
    };

    // the memory opens for the coming piece (or goes back to the end of the audio so far); past keep_s seconds of
    // audio it starts over from the last few seconds of pieces (the piece's length isn't known while hearing ahead)
    auto hear_begin = [&] {
        if (h_open && h_kept > keep_s - 3) {
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
        h_base_pos = h_snap_pos;
        h_base_conv = h_snap_conv;
        h_part = 0;
        h_begun = true;
    };
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
    // hearing ahead: while he is still talking, the open piece's sound so far goes in, the ears' last row held back
    auto open_samples = [&] { return (size_t) (got - finder.piece_start()); };
    auto ahead_rows = [&] {
        if (h_busy || !queue.empty() || !finder.has_sound()) return 0;
        const int final_rows = (int) (open_samples() / 1280) - 1;
        return final_rows - h_part >= 12 ? final_rows : 0;
    };
    auto hear_ahead_step = [&] {
        if (!h_begun) hear_begin();
        const size_t n = open_samples();
        std::vector<float> emb;
        const int P = ears.encode(audio.data() + (finder.piece_start() - a0), (int) n, emb);
        hear_rows(emb, h_part, P - 1);
    };
    auto hear_start = [&] {
        const piece & p = queue.front();
        if (!h_begun) hear_begin();
        h_P = ears.encode(p.pcm.data(), (int) p.pcm.size(), h_emb);
        if (h_part >= h_P) {   // the cut came before rows heard ahead: back to the piece's start
            h_kept -= h_part * 0.08;
            h_snap_pos = h_base_pos;
            h_snap_conv = h_base_conv;
            h_part = 0;
        }
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
            const std::vector<int32_t> nk = ko_ids(kept.size(), s.ko_in);
            brain.step_batch(nk.data(), nullptr, (int) nk.size(), false);
            x_snap_pos = brain.pos;
            x_snap_conv = brain.conv_state;
            std::vector<int32_t> given = tr_tail;
            for (auto & c : kept) given.insert(given.end(), c.en.begin(), c.en.end());
            brain.step_batch(given.data(), nullptr, (int) given.size(), true);
        });
        x_ids.clear();
        x_busy = true;
        if (voice_on && vp_sent < 0) vp_begin(x_sent);
    };
    auto hear_pick = [&] {
        const int w = best_of(H.logits);
        if (w != words.im_end && h_ids.size() < 512) { h_ids.push_back(w); h_word = w; return; }
        h_word = -1;
        h_busy = false;
        piece p = std::move(queue.front());
        queue.pop_front();
        heard.push_back({ h_emb, h_P, p.pcm.size() / 16000.0, trimmed(h_ids) });
        h_begun = false;
        h_part = 0;
        const std::string ko = imi_ko_piece(words.text(h_ids.data(), (int) h_ids.size()));
        heard_pieces++;
        if (!ko.empty() && ev.heard) ev.heard(ko);
        log("heard %.1f-%.1f s: %s", p.start, p.end, ko.c_str());
        ko_now_s += p.end - p.start;
        if (!ko.empty()) ko_now += (ko_now.empty() ? "" : " ") + ko;
        for (size_t e; (e = imi_ko_inner_end(ko_now)) != std::string::npos;) {   // a sentence end inside
            const double secs = ko_now_s * (double) e / ko_now.size();
            add_sentence(ko_now.substr(0, e), secs, p.end);
            ko_now = ko_now.substr(ko_now.find_first_not_of(' ', e));
            ko_now_s -= secs;
        }
        if (!ko_now.empty() && imi_ko_translate_now(ko_now, ko_now_s)) {
            add_sentence(ko_now, ko_now_s, p.end);
            ko_now.clear();
            ko_now_s = 0;
        }
    };
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
        if (w != words.im_end && x_ids.size() < 300) {
            x_ids.push_back(w);
            x_word = w;
            vp_update();
            if (ev.writing) {
                std::string t = words.text(x_ids.data(), (int) x_ids.size());
                const size_t a = t.find_first_not_of(' ');
                ev.writing((int) x_sent, a == std::string::npos ? "" : t.substr(a));
            }
            return;
        }
        x_word = -1;
        x_busy = false;
        sentence & s = sents[x_sent];
        const std::vector<int32_t> en = trimmed(x_ids);
        kept.push_back({ s.ko_in, en });
        s.en = words.text(en.data(), (int) en.size());
        while (!s.en.empty() && s.en.front() == ' ') s.en.erase(s.en.begin());
        sentences++;
        if (ev.said) ev.said((int) x_sent, s.ko, s.en);
        log("said %zu: %s -> %s", x_sent, s.ko.c_str(), s.en.c_str());
        if (voice_on && s.en.find_first_not_of(" \n") != std::string::npos) to_speak.push_back(x_sent);
        else if (vp_sent == (int) x_sent) vp_sent = -1;
    };
    auto speak_start = [&] {
        v_sent = to_speak.front();
        to_speak.pop_front();
        bool read = true;
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
            brain.swap_state(VP);   // VP's memory becomes the voice's; the old voice memory is free for the next one
            read = false;
        }
        vp_sent = -1; vp_bad = false; vp_plan.clear(); vp_done = 0;
        if (x_busy && voice_on) vp_begin(x_sent);
        tk.begin_sentences(words, en, 11 + (unsigned) v_sent, 2000, false, read);
        speaking = true;
        v_lead = true;
        v_held.clear();
    };
    auto voice_frame = [&] {
        const size_t before = pcm.size();
        speaking = tk.next(pcm, fb.data());
        sentence & s = sents[v_sent];
        const double made = now();
        std::vector<float> got_pcm(pcm.begin() + before, pcm.end());
        pcm.clear();
        if (!got_pcm.empty() && v_lead && cursor <= made) {
            // nothing is playing: the quiet start is held back, the sentence begins 60 ms before its first sound
            v_held.insert(v_held.end(), got_pcm.begin(), got_pcm.end());
            size_t first = v_held.size();
            for (size_t i = 0; i + 480 <= v_held.size(); i += 480) {
                double e = 0;
                for (size_t j = i; j < i + 480; j++) e += (double) v_held[j] * v_held[j];
                if (10 * std::log10(e / 480 + 1e-12) > -50) { first = i; break; }
            }
            const size_t keep = 1440;
            if (first == v_held.size()) {
                if (v_held.size() > keep) v_held.erase(v_held.begin(), v_held.end() - keep);
                got_pcm.clear();
            } else {
                got_pcm.assign(v_held.begin() + (first > keep ? first - keep : 0), v_held.end());
                v_held.clear();
                v_lead = false;
            }
        } else if (!got_pcm.empty()) v_lead = false;
        if (!got_pcm.empty()) {
            const double play = std::max(made, cursor);
            if (s.play_start < 0) s.play_start = play;
            cursor = play + got_pcm.size() / 24000.0;
            if (ev.speech) ev.speech((int) v_sent, got_pcm.data(), got_pcm.size());
        }
        if (!speaking) return;
        if (x_word >= 0) { brain.step2(-1, fb.data(), false, X, x_word, nullptr, true); translate_pick(); return; }
        if (h_word >= 0) { brain.step2(-1, fb.data(), false, H, h_word, nullptr, true); hear_pick(); return; }
        brain.step(-1, fb.data(), false);
    };
    auto hear_est = [&] {
        const piece & p = queue.front();
        int n = (int) (p.pcm.size() / 1280) - h_part + (int) post.size();
        for (auto & e : heard) n += (int) e.ids.size() + (h_open ? 0 : e.P);
        if (!h_open) n += (int) pre.size();
        return n * read_s + 0.035 * p.pcm.size() / 16000.0;
    };
    auto translate_est = [&] {
        int n = (int) tr_tail.size() + (int) sents[to_translate.front()].ko_in.size() / 3;
        for (auto & c : kept) n += (int) c.en.size() + (x_open ? 0 : (int) c.ko.size() / 3);
        return n * read_s;
    };

    log("listening");
    while (!stop) {
        // the sound that came in: cut at his pauses
        {
            std::vector<float> fresh;
            { std::lock_guard<std::mutex> l(mu); fresh.swap(inbox); }
            const bool on = listening;
            if (!on) {
                got += (int64_t) fresh.size();
                fresh.clear();
            } else if (!was_listening) {   // on again: the feed starts over here
                finder.reset(got);
                audio.clear();
                a0 = got;
            }
            was_listening = on;
            if (!fresh.empty()) {
                audio.insert(audio.end(), fresh.begin(), fresh.end());
                const std::vector<nunchi_pauses::cut> cuts = finder.push(fresh.data(), fresh.size());
                got += (int64_t) fresh.size();
                for (auto & c : cuts) {
                    piece p;
                    p.pcm.assign(audio.begin() + (c.start - a0), audio.begin() + (c.end - a0));
                    p.start = c.start / 16000.0;
                    p.end = c.end / 16000.0;
                    last_piece_end = p.end;
                    queue.push_back(std::move(p));
                }
                const int64_t keep_from = finder.piece_start();   // only the open piece's sound is kept
                if (keep_from > a0) { audio.erase(audio.begin(), audio.begin() + (keep_from - a0)); a0 = keep_from; }
            }
        }
        // he has gone quiet with a sentence unfinished: translate what there is
        if (!ko_now.empty() && queue.empty() && !h_busy && !finder.has_sound() && finder.quiet_s() >= 1.25) {
            add_sentence(ko_now, ko_now_s, last_piece_end);
            ko_now.clear();
            ko_now_s = 0;
        }
        const double t = now();
        const bool piece_in = !h_busy && !queue.empty();
        const bool tr_in = !x_busy && !to_translate.empty();
        const double ahead = speaking ? cursor - t : 0;
        if (!speaking && !to_speak.empty()) {
            speak_start();
        } else if (tr_in && (!speaking || ahead >= translate_est() + margin_s)) {
            translate_start(); translate_pick();
        } else if (piece_in && (!speaking || ahead >= hear_est() + margin_s)) {
            hear_start(); hear_pick();
        } else if (ahead_rows() > 0 && (!speaking || ahead >= 12 * read_s + 0.035 * open_samples() / 16000.0 + margin_s)) {
            hear_ahead_step();
        } else if (speaking) {
            voice_frame();
        } else if (x_word >= 0) {
            const int w = x_word;
            if (vp_sent == (int) x_sent && !vp_bad && vp_done < vp_plan.size()) {   // an English word rides into VP
                const int32_t vt = vp_plan[vp_done++];
                in(X, [&] { brain.step2(w, nullptr, true, VP, vt, nullptr, false); });
            } else in(X, [&] { brain.step(w, nullptr, true); });
            translate_pick();
        } else if (h_word >= 0) {
            const int w = h_word;
            in(H, [&] { brain.step(w, nullptr, true); });
            hear_pick();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    log("stopped");
}
